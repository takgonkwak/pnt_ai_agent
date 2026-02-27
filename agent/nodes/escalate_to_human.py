"""
Node: escalate_to_human
역할: 신뢰도 부족 또는 복잡한 문제 시 담당자에게 에스컬레이션
      FastMCP notify-server를 통해 Slack 알림 및 티켓 생성
"""
import logging
from agent.state import MESAgentState
from servers.notify_server import notify_mcp
from servers.mcp_utils import call_mcp

logger = logging.getLogger(__name__)


async def escalate_to_human(state: MESAgentState) -> MESAgentState:
    """에스컬레이션 노드"""
    logger.info(f"[escalate_to_human] session={state['session_id']}, level={state.get('escalation_level')}")

    escalation_level = state.get("escalation_level", "L1")
    summary = {
        "session_id": state.get("session_id"),
        "user_id": state.get("user_id"),
        "query": state.get("original_query", "")[:200],
        "category": state.get("query_category"),
        "confidence": state.get("answer_confidence", 0),
        "error_codes": state.get("error_codes", []),
        "is_program_issue": state.get("is_program_issue", False),
        "partial_answer": state.get("final_answer", "")[:300],
    }

    ticket_id = None
    try:
        await call_mcp(notify_mcp, "send_slack", level=escalation_level, summary=summary)

        ticket_id = await call_mcp(
            notify_mcp, "create_ticket",
            user_id=state.get("user_id"),
            session_id=state.get("session_id"),
            query=state.get("original_query", ""),
            category=state.get("query_category", "UNKNOWN"),
            analysis_summary=summary,
            escalation_level=escalation_level,
        )
        logger.info(f"[escalate_to_human] 티켓 생성: {ticket_id}")

    except Exception as e:
        logger.error(f"[escalate_to_human] 알림 오류: {e}")

    ticket_info = (
        f"\n\n---\n> 📋 **지원 티켓이 생성되었습니다** (티켓번호: {ticket_id})\n"
        f"> 담당자({escalation_level})가 확인 후 연락드립니다."
        if ticket_id else ""
    )

    return {
        **state,
        "final_answer": state.get("final_answer", "") + ticket_info,
        "processing_steps": state.get("processing_steps", []) + [
            f"🚨 에스컬레이션: {escalation_level}, 티켓={ticket_id}"
        ],
    }
