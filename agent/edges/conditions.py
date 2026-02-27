"""
LangGraph 조건부 엣지 (라우팅 로직)
"""
from typing import Literal
from agent.state import MESAgentState
from config import settings


def route_after_analysis(
    state: MESAgentState,
) -> Literal["clarify_user", "mes_query_executor"]:
    """
    질의 분석 후 라우팅:
    - 추가 정보 필요 + 재문의 횟수 미달 → 사용자 재문의
    - 그 외 → MES 조회 진행
    """
    missing = state.get("missing_info", [])
    count = state.get("clarification_count", 0)
    has_error = state.get("error_state")

    if has_error:
        # 분석 자체가 실패하면 바로 조회 진행 (최선 노력)
        return "mes_query_executor"

    if missing and count < settings.max_clarification_turns:
        return "clarify_user"

    return "mes_query_executor"


def route_after_log(
    state: MESAgentState,
) -> Literal["source_analyzer", "rag_retriever"]:
    """
    로그 조회 후 라우팅:
    - 프로그램 문제 + 관련 프로그램 있음 → 소스 분석
    - 그 외 → RAG 검색으로 바로 진행
    """
    if state.get("is_program_issue") and state.get("related_programs"):
        return "source_analyzer"
    return "rag_retriever"


def route_after_answer(
    state: MESAgentState,
) -> Literal["escalate_to_human", "__end__"]:
    """
    답변 생성 후 라우팅:
    - 에스컬레이션 필요 → 담당자 에스컬레이션
    - 그 외 → 종료
    """
    if state.get("escalation_required"):
        return "escalate_to_human"
    return "__end__"
