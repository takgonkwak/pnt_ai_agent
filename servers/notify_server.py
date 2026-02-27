"""
FastMCP Server: notify-server
에스컬레이션 알림 (Slack) 및 티켓 생성
"""
import uuid
import logging
from datetime import datetime
from typing import Optional
from fastmcp import FastMCP
from config import settings

logger = logging.getLogger(__name__)

notify_mcp = FastMCP("notify-server")

ESCALATION_CHANNELS = {"L1": "#mes-support", "L2": "#mes-dev-team", "L3": "#mes-dba-infra"}

# 인메모리 티켓 저장소 (운영: Jira / ServiceNow 연동)
_ticket_store: dict = {}

_use_mock = not bool(settings.slack_bot_token)


@notify_mcp.tool()
async def send_slack(level: str, summary: dict) -> bool:
    """Slack 에스컬레이션 알림 전송 (L1/L2/L3 채널)"""
    channel = ESCALATION_CHANNELS.get(level, "#mes-support")
    emoji = {"L1": "⚠️", "L2": "🔧", "L3": "🚨"}.get(level, "📋")
    message = (
        f"{emoji} *MES AI Agent 에스컬레이션 [{level}]*\n"
        f"• 세션: `{summary.get('session_id', 'N/A')}`\n"
        f"• 사용자: `{summary.get('user_id', 'N/A')}`\n"
        f"• 카테고리: `{summary.get('category', 'N/A')}`\n"
        f"• 신뢰도: `{summary.get('confidence', 0):.1%}`\n"
        f"• 에러코드: `{', '.join(summary.get('error_codes', [])) or '없음'}`\n"
        f"• 질의: _{summary.get('query', '')[:100]}_"
    )

    if _use_mock:
        logger.info(f"[notify][MOCK] Slack → {channel}\n{message}")
        return True

    try:
        import httpx
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://slack.com/api/chat.postMessage",
                headers={"Authorization": f"Bearer {settings.slack_bot_token}"},
                json={"channel": channel, "text": message},
                timeout=10.0,
            )
            ok = resp.json().get("ok", False)
            logger.info(f"[notify] Slack 전송: ok={ok}")
            return ok
    except Exception as e:
        logger.error(f"[notify] Slack 오류: {e}")
        return False


@notify_mcp.tool()
async def create_ticket(
    user_id: str,
    session_id: str,
    query: str,
    category: str,
    analysis_summary: dict,
    escalation_level: str,
) -> str:
    """지원 티켓 생성 (인메모리 / 운영: Jira·ServiceNow 연동)"""
    ticket_id = f"MES-{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:6].upper()}"
    _ticket_store[ticket_id] = {
        "ticket_id": ticket_id,
        "created_at": datetime.now().isoformat(),
        "user_id": user_id,
        "session_id": session_id,
        "query": query,
        "category": category,
        "escalation_level": escalation_level,
        "status": "OPEN",
        "analysis": analysis_summary,
    }
    logger.info(f"[notify] 티켓 생성: {ticket_id}")
    return ticket_id


# ── 티켓 조회 헬퍼 (API 레이어용) ───────────────────────

def get_ticket(ticket_id: str) -> Optional[dict]:
    return _ticket_store.get(ticket_id)


def list_open_tickets() -> list:
    return [t for t in _ticket_store.values() if t.get("status") == "OPEN"]
