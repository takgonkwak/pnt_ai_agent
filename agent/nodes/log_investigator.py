"""
Node 3: log_investigator
역할: 시스템 로그에서 문의 관련 이벤트/에러 조사
      FastMCP log-server를 통해 로그 검색 및 프로그램 문제 여부 판단
"""
import re
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from agent.state import MESAgentState
from config import settings
from servers.log_server import log_mcp
from servers.mcp_utils import call_mcp

logger = logging.getLogger(__name__)

STACK_TRACE_PATTERN = re.compile(
    r'at\s+([\w\.]+)\.([\w\$]+)\.([\w\$<>]+)\((\w+\.java):(\d+)\)'
)
ERROR_CODE_PATTERN = re.compile(r'\[([A-Z]{2,6}-\d{3,6})\]')


def _build_time_range(extracted_info: Dict[str, Any], window_minutes: int) -> tuple[str, str]:
    ts = extracted_info.get("timestamp")
    if ts:
        try:
            center = datetime.fromisoformat(ts)
        except ValueError:
            center = datetime.now()
    else:
        center = datetime.now()
    return (
        (center - timedelta(minutes=window_minutes)).isoformat(),
        (center + timedelta(minutes=window_minutes)).isoformat(),
    )


def _extract_program_references(log_entries: List[Dict]) -> List[Dict[str, Any]]:
    programs, seen = [], set()
    for entry in log_entries:
        stack_trace = entry.get("stack_trace", "")
        for match in STACK_TRACE_PATTERN.finditer(stack_trace):
            pkg, cls, method, fname, lno = match.groups()
            key = f"{cls}.{method}:{lno}"
            if key not in seen:
                seen.add(key)
                programs.append({
                    "package": pkg, "class_name": cls, "method_name": method,
                    "file_name": fname, "line_number": int(lno),
                    "error_message": entry.get("message", "")[:200],
                    "stack_trace": stack_trace[:1000],
                })
    return programs


def _extract_error_codes(log_entries: List[Dict]) -> List[str]:
    codes = set()
    for entry in log_entries:
        for m in ERROR_CODE_PATTERN.finditer(entry.get("message", "")):
            codes.add(m.group(1))
    return list(codes)


async def log_investigator(state: MESAgentState) -> MESAgentState:
    """로그 조회 노드"""
    logger.info(f"[log_investigator] session={state['session_id']}")

    extracted = state.get("extracted_info", {})
    from_time, to_time = _build_time_range(extracted, settings.log_search_window_minutes)

    keywords = []
    for key in ("lot_id", "equip_id", "error_message", "error_code"):
        if extracted.get(key):
            keywords.append(extracted[key])
    if not keywords:
        keywords.append(state.get("query_intent", "")[:50])

    try:
        half = settings.log_max_search_results // 2

        error_logs = await call_mcp(
            log_mcp, "search_logs",
            keywords=keywords, from_time=from_time, to_time=to_time,
            log_level="ERROR", user_id=extracted.get("user_id"), max_results=half,
        ) or []

        warn_logs = await call_mcp(
            log_mcp, "search_logs",
            keywords=keywords, from_time=from_time, to_time=to_time,
            log_level="WARN", user_id=extracted.get("user_id"), max_results=half,
        ) or []

        all_logs = sorted(
            error_logs + warn_logs,
            key=lambda x: x.get("timestamp", ""),
            reverse=True,
        )[:settings.log_max_search_results]

        related_programs = _extract_program_references(all_logs)
        error_codes = _extract_error_codes(all_logs)
        is_program_issue = (
            len([e for e in all_logs if e.get("level") == "ERROR"]) > 0
            and len(related_programs) > 0
        )

        logger.info(
            f"[log_investigator] 로그={len(all_logs)}건, "
            f"프로그램={len(related_programs)}건, 에러코드={error_codes}"
        )

        return {
            **state,
            "log_entries": all_logs,
            "error_codes": error_codes,
            "related_programs": related_programs,
            "is_program_issue": is_program_issue,
            "processing_steps": state.get("processing_steps", []) + [
                f"✅ 로그 조회 완료: {len(all_logs)}건, 에러코드={error_codes}, 프로그램이슈={is_program_issue}"
            ],
        }

    except Exception as e:
        logger.error(f"[log_investigator] 오류: {e}")
        return {
            **state,
            "log_entries": [], "error_codes": [], "related_programs": [],
            "is_program_issue": False,
            "processing_steps": state.get("processing_steps", []) + [
                f"⚠️ 로그 조회 오류: {str(e)}"
            ],
        }
