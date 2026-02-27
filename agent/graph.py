"""
LangGraph 메인 워크플로우 그래프 정의
MES AI Agent의 5단계 처리 흐름을 StateGraph로 구성
"""
import logging
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from agent.state import MESAgentState
from agent.nodes.query_analyzer import query_analyzer
from agent.nodes.clarify_user import clarify_user
from agent.nodes.mes_query_executor import mes_query_executor
from agent.nodes.log_investigator import log_investigator
from agent.nodes.source_analyzer import source_analyzer
from agent.nodes.rag_retriever import rag_retriever
from agent.nodes.answer_generator import answer_generator
from agent.nodes.escalate_to_human import escalate_to_human
from agent.edges.conditions import (
    route_after_analysis,
    route_after_log,
    route_after_answer,
)

logger = logging.getLogger(__name__)


def build_graph() -> StateGraph:
    """
    LangGraph StateGraph 빌드 및 반환

    흐름:
    START → query_analyzer
          → (조건) clarify_user ──interrupt()──→ [사용자 답변 대기]
                                ──resume──→ query_analyzer (재분석)
          → mes_query_executor   (MES 조회)
          → log_investigator     (로그 조회)
          → (조건) source_analyzer (프로그램 이슈 시)
          → rag_retriever
          → answer_generator
          → (조건) escalate_to_human
          → END
    """
    graph = StateGraph(MESAgentState)

    # ── 노드 등록 ────────────────────────────────────────
    graph.add_node("query_analyzer", query_analyzer)
    graph.add_node("clarify_user", clarify_user)
    graph.add_node("mes_query_executor", mes_query_executor)
    graph.add_node("log_investigator", log_investigator)
    graph.add_node("source_analyzer", source_analyzer)
    graph.add_node("rag_retriever", rag_retriever)
    graph.add_node("answer_generator", answer_generator)
    graph.add_node("escalate_to_human", escalate_to_human)

    # ── 엣지 연결 ────────────────────────────────────────
    # 시작
    graph.set_entry_point("query_analyzer")

    # 질의 분석 후: 재문의 OR MES 조회
    graph.add_conditional_edges(
        "query_analyzer",
        route_after_analysis,
        {
            "clarify_user": "clarify_user",
            "mes_query_executor": "mes_query_executor",
        },
    )

    # 재문의 후: interrupt() 재개 후 → 재분석 (enriched_query로 다시 분석)
    graph.add_edge("clarify_user", "query_analyzer")

    # MES 조회 완료 후: 로그 조회
    graph.add_edge("mes_query_executor", "log_investigator")

    # 로그 조회 후: 소스 분석 OR RAG 검색
    graph.add_conditional_edges(
        "log_investigator",
        route_after_log,
        {
            "source_analyzer": "source_analyzer",
            "rag_retriever": "rag_retriever",
        },
    )

    # 소스 분석 후: 항상 RAG 검색
    graph.add_edge("source_analyzer", "rag_retriever")

    # RAG 검색 후: 항상 답변 생성
    graph.add_edge("rag_retriever", "answer_generator")

    # 답변 생성 후: 에스컬레이션 OR 종료
    graph.add_conditional_edges(
        "answer_generator",
        route_after_answer,
        {
            "escalate_to_human": "escalate_to_human",
            "__end__": END,
        },
    )

    # 에스컬레이션 후: 종료
    graph.add_edge("escalate_to_human", END)

    return graph


# 컴파일된 그래프 & 체크포인터 (싱글톤)
_compiled_graph = None
_checkpointer = None


def get_graph():
    """컴파일된 LangGraph 반환 (싱글톤, MemorySaver 체크포인터 포함)"""
    global _compiled_graph, _checkpointer
    if _compiled_graph is None:
        graph = build_graph()
        _checkpointer = MemorySaver()
        _compiled_graph = graph.compile(checkpointer=_checkpointer)
        logger.info("[graph] LangGraph 컴파일 완료 (interrupt 지원)")
    return _compiled_graph
