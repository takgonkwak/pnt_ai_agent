"""
FastMCP Server: log-server
MES 시스템 로그 검색 및 스택 트레이스 조회

실제 환경: Elasticsearch (ES_ENABLED=true)
개발/테스트: 샘플 로그 데이터 (ES_ENABLED=false)

ES 인덱스 필드 규약 (Logstash/ECS 공통):
  @timestamp   - ISO8601 (e.g. 2024-01-15T14:30:05.000Z)
  level        - 로그 레벨 (ERROR | WARN | INFO | DEBUG)
  logger_name  - 클래스 풀네임 (e.g. com.mes.equipment.WireBondService)
  method_name  - 메서드명
  line_number  - 라인 번호 (integer)
  message      - 로그 메시지
  stack_trace  - 예외 스택 트레이스 (없으면 빈 문자열 또는 필드 없음)
  mdc.user_id  - MDC: 사용자 ID
  mdc.session_id - MDC: 세션 ID
  mdc.lot_id   - MDC: Lot ID
"""
import logging
from typing import Optional
from fastmcp import FastMCP
from config import settings

from servers.mcp_utils import load_mock_data

logger = logging.getLogger(__name__)

log_mcp = FastMCP("log-server")

# Mock 데이터 로드
_mock_data = load_mock_data()
SAMPLE_LOGS = _mock_data.get("logs", [])

_LEVEL_PRIORITY = {"ERROR": 0, "WARN": 1, "INFO": 2, "DEBUG": 3}


# ── MCP Tools ────────────────────────────────────────────

@log_mcp.tool()
async def search_logs(
    keywords: list,
    from_time: str,
    to_time: str,
    log_level: str = "ERROR",
    user_id: Optional[str] = None,
    max_results: int = 50,
) -> list:
    """
    키워드, 시간 범위, 레벨로 MES 시스템 로그 검색 (Elasticsearch)

    Args:
        keywords:    검색 키워드 목록 (message, stack_trace, class_name 대상)
        from_time:   검색 시작 시각 (ISO8601 또는 'now-1h' 상대표현)
        to_time:     검색 종료 시각 (ISO8601 또는 'now')
        log_level:   최소 로그 레벨 (ERROR | WARN | INFO | DEBUG)
        user_id:     특정 사용자 필터 (MDC 필드)
        max_results: 최대 반환 건수
    """
    logger.info(f"[log] search_logs: keywords={keywords}, level={log_level}, "
                f"from={from_time}, to={to_time}")

    if not settings.es_enabled:
        return _search_mock(keywords, log_level, user_id, max_results)

    return await _search_es(keywords, from_time, to_time, log_level, user_id, max_results)


@log_mcp.tool()
async def get_stack_trace(
    error_id: str,
    session_id: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> Optional[str]:
    """
    특정 에러의 스택 트레이스 조회

    Args:
        error_id:   ES document _id 또는 에러 correlation_id
        session_id: 세션 ID (조회 범위 보조)
        timestamp:  에러 발생 시각 (ISO8601, 검색 범위 ±5분)
    """
    if not settings.es_enabled:
        for log in SAMPLE_LOGS:
            if log.get("level") == "ERROR" and log.get("stack_trace"):
                return log["stack_trace"]
        return None

    return await _get_stack_trace_es(error_id, session_id, timestamp)


# ── Mock 헬퍼 ────────────────────────────────────────────

def _search_mock(keywords, log_level, user_id, max_results):
    target_p = _LEVEL_PRIORITY.get(log_level, 2)
    results = []
    for log in SAMPLE_LOGS:
        if _LEVEL_PRIORITY.get(log.get("level", "INFO"), 2) > target_p:
            continue
        if user_id and log.get("user_id") != user_id:
            continue
        log_text = " ".join([
            log.get("message", ""),
            log.get("stack_trace", ""),
            log.get("class_name", ""),
        ]).lower()
        if not keywords or any(str(kw).lower() in log_text for kw in keywords):
            results.append(dict(log))
    return results[:max_results]


# ── Elasticsearch 헬퍼 ───────────────────────────────────

def _build_es_client():
    """AsyncElasticsearch 클라이언트 생성"""
    from elasticsearch import AsyncElasticsearch

    kwargs = {
        "hosts": [{"host": settings.es_host, "port": settings.es_port,
                   "scheme": settings.es_scheme}],
    }
    if settings.es_user and settings.es_password:
        kwargs["http_auth"] = (settings.es_user, settings.es_password)

    return AsyncElasticsearch(**kwargs)


def _build_search_query(keywords, from_time, to_time, log_level, user_id, max_results):
    """
    ES bool query 생성

    - filter: 시간 범위 + 레벨 (캐시 효율 good)
    - should: 키워드 multi_match (message, stack_trace, logger_name)
    """
    # 지정 레벨 이상의 레벨만 포함 (ERROR → [ERROR], WARN → [ERROR, WARN], ...)
    priority = _LEVEL_PRIORITY.get(log_level.upper(), 2)
    included_levels = [lvl for lvl, p in _LEVEL_PRIORITY.items() if p <= priority]

    filters = [
        {"range": {"@timestamp": {"gte": from_time, "lte": to_time}}},
        {"terms": {"level.keyword": included_levels}},
    ]

    if user_id:
        filters.append({"term": {"mdc.user_id.keyword": user_id}})

    query: dict = {"bool": {"filter": filters}}

    if keywords:
        keyword_str = " ".join(str(k) for k in keywords)
        query["bool"]["must"] = [
            {
                "multi_match": {
                    "query": keyword_str,
                    "fields": ["message", "stack_trace", "logger_name"],
                    "type": "best_fields",
                    "operator": "or",
                }
            }
        ]

    return {
        "query": query,
        "sort": [{"@timestamp": {"order": "desc"}}],
        "size": max_results,
        "_source": [
            "@timestamp", "level", "logger_name",
            "method_name", "line_number",
            "message", "stack_trace",
            "mdc.user_id", "mdc.session_id",
        ],
    }


def _hit_to_log_entry(hit: dict) -> dict:
    """ES hit._source → 공통 로그 엔트리 딕셔너리 변환"""
    src = hit.get("_source", {})
    mdc = src.get("mdc", {})
    logger_name = src.get("logger_name", "")
    class_name = logger_name.split(".")[-1] if logger_name else ""
    return {
        "es_id": hit.get("_id", ""),
        "timestamp": src.get("@timestamp", ""),
        "level": src.get("level", ""),
        "class_name": class_name,
        "logger_name": logger_name,
        "method_name": src.get("method_name", ""),
        "line_number": src.get("line_number"),
        "message": src.get("message", ""),
        "stack_trace": src.get("stack_trace", ""),
        "user_id": mdc.get("user_id", ""),
        "session_id": mdc.get("session_id", ""),
    }


async def _search_es(keywords, from_time, to_time, log_level, user_id, max_results) -> list:
    es = _build_es_client()
    try:
        body = _build_search_query(keywords, from_time, to_time, log_level, user_id, max_results)
        resp = await es.search(index=settings.es_index_pattern, body=body)
        hits = resp.get("hits", {}).get("hits", [])
        return [_hit_to_log_entry(h) for h in hits]
    except Exception as e:
        logger.error(f"[log] ES search 오류: {e}")
        return []
    finally:
        await es.close()


async def _get_stack_trace_es(error_id: str, session_id: Optional[str],
                               timestamp: Optional[str]) -> Optional[str]:
    """
    ES에서 스택 트레이스 조회
    1순위: _id로 직접 GET
    2순위: timestamp ±5분 범위에서 session_id + ERROR 레벨 검색
    """
    es = _build_es_client()
    try:
        # 1순위: document ID로 직접 조회
        try:
            doc = await es.get(index=settings.es_index_pattern, id=error_id)
            return doc["_source"].get("stack_trace")
        except Exception:
            pass

        # 2순위: 시간 범위 + 세션으로 검색
        if not timestamp:
            return None

        filters: list = [
            {"range": {"@timestamp": {"gte": f"{timestamp}||-5m",
                                      "lte": f"{timestamp}||+5m"}}},
            {"term": {"level.keyword": "ERROR"}},
            {"exists": {"field": "stack_trace"}},
        ]
        if session_id:
            filters.append({"term": {"mdc.session_id.keyword": session_id}})

        body = {
            "query": {"bool": {"filter": filters}},
            "sort": [{"@timestamp": {"order": "desc"}}],
            "size": 1,
            "_source": ["stack_trace"],
        }
        resp = await es.search(index=settings.es_index_pattern, body=body)
        hits = resp.get("hits", {}).get("hits", [])
        if hits:
            return hits[0]["_source"].get("stack_trace")
        return None

    except Exception as e:
        logger.error(f"[log] ES get_stack_trace 오류: {e}")
        return None
    finally:
        await es.close()
