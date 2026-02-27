"""
Node: clarify_user
역할: 질의 분석 결과 추가 정보가 필요할 때 사용자에게 재문의
      LangGraph interrupt()로 그래프를 일시 정지 → Human-in-the-Loop

흐름:
  1. 재문의 질문 생성
  2. interrupt({"question": ...}) 호출 → 그래프 일시 정지, 체크포인터에 상태 저장
  3. 클라이언트가 답변 후 Command(resume=answer) 로 재개
  4. 사용자 답변을 original_query에 추가하고 query_analyzer로 복귀 (재분석)
"""
import logging
from langgraph.types import interrupt
from agent.state import MESAgentState

logger = logging.getLogger(__name__)


async def clarify_user(state: MESAgentState) -> MESAgentState:
    """
    사용자 재문의 노드 - Human-in-the-Loop

    interrupt() 호출 시 그래프 실행이 일시 정지되고, API/WebSocket 레이어에서
    클라이언트에 질문을 전달합니다. Command(resume=answer)로 재개하면
    interrupt() 반환값으로 사용자의 답변을 받아 원본 질의를 보강합니다.
    """
    count = state.get("clarification_count", 0) + 1
    logger.info(f"[clarify_user] session={state['session_id']} turn={count}")

    clarification_q = state.get("clarification_question", "")
    if not clarification_q:
        missing = state.get("missing_info", [])
        clarification_q = (
            "문의 처리를 위해 추가 정보가 필요합니다. "
            f"다음 정보를 제공해 주세요: {', '.join(missing)}"
        )

    # ── Human-in-the-Loop: 그래프 일시 정지 ──────────────────────────
    # interrupt() 호출 시 그래프 실행 중단, 체크포인터에 현재 상태 저장
    # API: await graph.ainvoke(...) 가 여기서 조기 반환됨
    # 재개: await graph.ainvoke(Command(resume=user_answer), config=config)
    # interrupt() 반환값 = Command(resume=...) 에 전달한 값
    human_reply = interrupt({"question": clarification_q})
    # ─────────────────────────────────────────────────────────────────

    # 사용자 답변을 원본 질의에 추가하여 재분석 준비
    enriched_query = f"{state['original_query']}\n[추가 정보] {human_reply}"
    logger.info(f"[clarify_user] 답변 수신 → 재분석 진행: {str(human_reply)[:60]}")

    return {
        **state,
        "original_query": enriched_query,
        "clarification_count": count,
        "missing_info": [],             # 재분석 전 초기화
        "clarification_question": "",   # 재분석 전 초기화
        "processing_steps": state.get("processing_steps", []) + [
            f"🔄 재문의 완료 ({count}회차): {str(human_reply)[:60]}"
        ],
    }
