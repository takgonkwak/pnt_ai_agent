"""
Node: rag_retriever
역할: 벡터 DB에서 유사 사례 및 MES 매뉴얼 관련 문서 검색
      answer_generator 노드 실행 전에 항상 실행
"""
import logging
from agent.state import MESAgentState
from rag.retriever import MESRAGRetriever

logger = logging.getLogger(__name__)


async def rag_retriever(state: MESAgentState) -> MESAgentState:
    """RAG 검색 노드"""
    logger.info(f"[rag_retriever] session={state['session_id']}")

    retriever = MESRAGRetriever()

    # 검색 쿼리 구성 (질의 + 에러코드 + 카테고리)
    query_parts = [state.get("original_query", "")]
    if state.get("error_codes"):
        query_parts.extend(state["error_codes"])
    if state.get("root_cause"):
        query_parts.append(state["root_cause"])

    search_query = " ".join(filter(None, query_parts))

    context = {
        "query_category": state.get("query_category", "UNKNOWN"),
        "error_codes": state.get("error_codes", []),
        "extracted_info": state.get("extracted_info", {}),
    }

    try:
        # 1. 관련 문서 검색 (매뉴얼, 기술문서)
        rag_docs = await retriever.retrieve(
            query=search_query,
            context=context,
            k=5,
        )

        # 2. 유사 장애 사례 검색
        similar = await retriever.retrieve(
            query=search_query,
            context={**context, "category_filter": "incident"},
            k=3,
        )

        logger.info(
            f"[rag_retriever] 문서={len(rag_docs)}건, 유사사례={len(similar)}건"
        )

        return {
            **state,
            "rag_context": [doc.__dict__ for doc in rag_docs],
            "similar_cases": [doc.__dict__ for doc in similar],
            "processing_steps": state.get("processing_steps", []) + [
                f"✅ RAG 검색 완료: 관련문서 {len(rag_docs)}건, 유사사례 {len(similar)}건"
            ],
        }

    except Exception as e:
        logger.error(f"[rag_retriever] 오류: {e}")
        return {
            **state,
            "rag_context": [],
            "similar_cases": [],
            "processing_steps": state.get("processing_steps", []) + [
                f"⚠️ RAG 검색 오류: {str(e)}"
            ],
        }
