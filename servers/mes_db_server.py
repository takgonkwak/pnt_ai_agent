"""
FastMCP Server: mes-db-server
MES DB 조회 도구 (Read-Only)
실제 환경: Oracle/MSSQL 연결, 개발/테스트: Mock 데이터
"""
import logging
from typing import Optional
from fastmcp import FastMCP
from config import settings

from servers.mcp_utils import load_mock_data
from servers.sql_queries import QUERY_LOT, QUERY_EQUIPMENT, QUERY_WIP_BASE

logger = logging.getLogger(__name__)

mes_db_mcp = FastMCP("mes-db-server")

# Mock 데이터 로드
_mock_data = load_mock_data()
SAMPLE_LOT_DATA = _mock_data.get("lot_data", {})
SAMPLE_EQUIP_DATA = _mock_data.get("equip_data", {})

_use_mock = not bool(settings.mes_db_user)


# ── FastMCP 도구 정의 ─────────────────────────────────────

@mes_db_mcp.tool()
async def query_lot(lot_id: str, include_history: bool = False) -> dict:
    """
    특정 Lot 번호를 사용하여 MES에서 현재 Lot의 상세 정보(상태, 위치, 현재공정, 제품, 수량 등)를 조회합니다.
    사용자가 특정 작업(Lot)의 지연 사유나 현재 어디에 있는지를 물어볼 때 사용합니다.
    
    Args:
        lot_id: 조회할 Lot의 고유 번호 (예: 'LOT-2024-001')
        include_history: True일 경우 해당 Lot의 과거 공정 이동 이력을 포함하여 상세히 조회합니다.
    """
    logger.info(f"[mes_db] query_lot: {lot_id}")

    if _use_mock:
        data = SAMPLE_LOT_DATA.get(lot_id)
        if not data:
            return {"lot_id": lot_id, "status": "NOT_FOUND",
                    "message": f"Lot {lot_id}을 찾을 수 없습니다."}
        result = dict(data)
        if not include_history:
            result.pop("history", None)
        return result

    return await _db_query(
        QUERY_LOT,
        {"lot_id": lot_id},
    )


@mes_db_mcp.tool()
async def query_wip(
    equip_id: Optional[str] = None,
    process_id: Optional[str] = None,
    product_id: Optional[str] = None,
    status: Optional[str] = None,
) -> list:
    """
    WIP(Work In Process, 재공) 현황을 조회합니다. 
    특정 설비에 쌓인 Lot 목록, 특정 공정의 대기 물량, 또는 특정 제품의 생산 현황 등을 파악할 때 사용합니다.
    아무 인자도 주지 않으면 전체 재공 목록을 최대 100건까지 반환합니다.

    Args:
        equip_id: 특정 설비(Resource)에 할당된 재공만 필터링할 때 사용합니다.
        process_id: 특정 공정 단계에 있는 재공만 필터링할 때 사용합니다.
        product_id: 특정 제품(Item)에 해당하는 재공만 필터링할 때 사용합니다.
        status: 특정 상태(예: 'RUN', 'HOLD', 'WAIT')인 재공만 필터링할 때 사용합니다.
    """
    logger.info(f"[mes_db] query_wip: equip={equip_id}, process={process_id}")

    if _use_mock:
        results = []
        for lot in SAMPLE_LOT_DATA.values():
            if equip_id and lot["equip_id"] != equip_id:
                continue
            if process_id and lot["process_id"] != process_id:
                continue
            if product_id and lot["product_id"] != product_id:
                continue
            if status and lot["status"] != status:
                continue
            results.append({k: lot[k] for k in
                             ("lot_id", "product_id", "process_id", "equip_id", "status", "qty")})
        return results or [{"message": "조건에 해당하는 WIP 없음"}]

    conditions, params = [], {}
    if equip_id:
        conditions.append("EQUIP_ID = :equip_id"); params["equip_id"] = equip_id
    if process_id:
        conditions.append("PROCESS_ID = :process_id"); params["process_id"] = process_id
    if product_id:
        conditions.append("PRODUCT_ID = :product_id"); params["product_id"] = product_id
    if status:
        conditions.append("STATUS = :status"); params["status"] = status
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    # ROWNUM 제한은 Oracle 기준 (타 DB는 LIMIT 등 사용)
    sql = f"{QUERY_WIP_BASE} {where} AND ROWNUM <= 100"
    return await _db_query(sql, params)


@mes_db_mcp.tool()
async def query_equipment(equip_id: str) -> dict:
    """
    특정 설비(Equipment)의 실시간 상태 정보를 조회합니다.
    설비의 가동 여부(RUN/DOWN/ALARM), 현재 진행 중인 레시피, 발생한 알람 메시지, 마지막 예방정비(PM) 일자 등을 확인할 때 사용합니다.

    Args:
        equip_id: 조회할 설비의 고유 ID (예: 'EQUIP-WB01')
    """
    logger.info(f"[mes_db] query_equipment: {equip_id}")

    if _use_mock:
        data = SAMPLE_EQUIP_DATA.get(equip_id)
        return data or {"equip_id": equip_id, "status": "NOT_FOUND"}

    return await _db_query(
        QUERY_EQUIPMENT,
        {"equip_id": equip_id},
    )


@mes_db_mcp.tool()
async def run_sql(sql: str, max_rows: int = 100) -> list:
    """MES DB에 읽기 전용 SELECT 쿼리 직접 실행 (DML 차단)"""
    sql_upper = sql.strip().upper()
    if not sql_upper.startswith("SELECT"):
        raise PermissionError("SELECT 쿼리만 허용됩니다.")
    for kw in ("INSERT", "UPDATE", "DELETE", "DROP", "TRUNCATE", "ALTER", "CREATE"):
        if kw in sql_upper:
            raise PermissionError(f"'{kw}' 문은 허용되지 않습니다.")

    logger.info(f"[mes_db] run_sql: {sql[:80]}...")
    if _use_mock:
        return [{"mock": "DB 미연결 - Mock 모드", "sql": sql[:50]}]
    return await _db_query(sql, {}, max_rows=max_rows)


# ── 내부 DB 실행 헬퍼 ────────────────────────────────────

async def _db_query(sql: str, params: dict, max_rows: int = 1000):
    try:
        from sqlalchemy.ext.asyncio import create_async_engine
        from sqlalchemy import text
        engine = create_async_engine(_build_url(), echo=False)
        async with engine.connect() as conn:
            result = await conn.execute(text(sql), params)
            rows = result.fetchmany(max_rows)
            columns = list(result.keys())
            return [dict(zip(columns, row)) for row in rows]
    except Exception as e:
        logger.error(f"[mes_db] DB 오류: {e}")
        return {"error": str(e)}


def _build_url() -> str:
    t = settings.mes_db_type.lower()
    u, p, h, port, n = (settings.mes_db_user, settings.mes_db_password,
                        settings.mes_db_host, settings.mes_db_port, settings.mes_db_name)
    if t == "oracle":
        return f"oracle+cx_oracle://{u}:{p}@{h}:{port}/{n}"
    if t == "mssql":
        return (f"mssql+aioodbc://{u}:{p}@{h}/{n}"
                "?driver=ODBC+Driver+17+for+SQL+Server")
    if t == "postgresql":
        return f"postgresql+asyncpg://{u}:{p}@{h}:{port}/{n}"
    raise ValueError(f"지원하지 않는 DB 타입: {t}")
