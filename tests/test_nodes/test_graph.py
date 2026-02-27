"""
LangGraph Agent 통합 테스트 (Mock 모드)
실제 LLM 호출 없이 Mock으로 전체 흐름 검증
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from agent.state import create_initial_state, MESAgentState


def make_mock_llm_response(content: str):
    mock_response = MagicMock()
    mock_response.content = content
    return mock_response


MOCK_ANALYSIS_RESPONSE = """{
    "query_intent": "LOT-2024-001 HOLD 원인 파악",
    "query_category": "DATA_ERROR",
    "extracted_info": {
        "lot_id": "LOT-2024-001",
        "equip_id": "EQUIP-WB01",
        "process_id": null,
        "product_id": null,
        "user_id": null,
        "screen_name": null,
        "error_message": null,
        "error_code": null,
        "timestamp": null,
        "additional": {}
    },
    "missing_info": [],
    "clarification_question": "",
    "screenshot_findings": ""
}"""

MOCK_ANSWER_RESPONSE = """{
    "answer": "## 현황 요약\nLOT-2024-001은 와이어본딩 공정에서 HOLD 상태입니다.\n\n## 원인 분석\n와이어 장력 초과 알람으로 HOLD 처리되었습니다.",
    "confidence": 0.85,
    "escalation_required": false,
    "escalation_level": "NONE",
    "escalation_reason": ""
}"""


@pytest.mark.asyncio
async def test_initial_state_creation():
    """초기 State 생성 테스트"""
    state = create_initial_state(
        user_id="test-user",
        session_id="test-session-001",
        query="LOT-2024-001이 왜 HOLD 됐나요?",
    )
    assert state["user_id"] == "test-user"
    assert state["session_id"] == "test-session-001"
    assert state["clarification_count"] == 0
    assert state["is_program_issue"] == False
    assert state["processing_steps"] == []


@pytest.mark.asyncio
async def test_query_analyzer_node():
    """query_analyzer 노드 단위 테스트 (ChatOpenAI Mock)"""
    from agent.nodes.query_analyzer import query_analyzer

    state = create_initial_state(
        user_id="user001",
        session_id="sess-001",
        query="LOT-2024-001이 왜 HOLD 됐나요?",
    )

    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(return_value=make_mock_llm_response(MOCK_ANALYSIS_RESPONSE))

    with patch("agent.nodes.query_analyzer.ChatOpenAI", return_value=mock_llm):
        result = await query_analyzer(state)

    assert result["query_category"] == "DATA_ERROR"
    assert result["extracted_info"]["lot_id"] == "LOT-2024-001"
    assert result["missing_info"] == []
    assert len(result["processing_steps"]) > 0


@pytest.mark.asyncio
async def test_mes_query_executor_mock():
    """mes_query_executor - FastMCP Mock DB 테스트 (LLM Tool Calling 대응)"""
    from agent.nodes.mes_query_executor import mes_query_executor

    state = create_initial_state(
        user_id="user001",
        session_id="sess-001",
        query="LOT-2024-001 HOLD 원인",
    )
    state["query_intent"] = "LOT HOLD 원인 파악"
    state["query_category"] = "DATA_ERROR"
    state["extracted_info"] = {"lot_id": "LOT-2024-001", "equip_id": "EQUIP-WB01"}

    # Tool calling 결과 시뮬레이션
    mock_llm = AsyncMock()
    mock_response = MagicMock()
    mock_response.tool_calls = [
        {"name": "query_lot", "args": {"lot_id": "LOT-2024-001", "include_history": True}}
    ]
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)
    mock_llm.bind_tools = MagicMock(return_value=mock_llm)

    with patch("agent.nodes.mes_query_executor.ChatOpenAI", return_value=mock_llm):
        result = await mes_query_executor(state)

    assert "mes_query_results" in result
    # 리팩토링 후 도구별 결과 키는 {tool_name}_result 형식이 됨
    assert "query_lot_result" in result["mes_query_results"]
    assert result["mes_query_results"]["query_lot_result"]["status"] == "HOLD"


@pytest.mark.asyncio
async def test_log_investigator_mock():
    """log_investigator - FastMCP Mock 로그 테스트"""
    from agent.nodes.log_investigator import log_investigator

    state = create_initial_state(
        user_id="user001",
        session_id="sess-001",
        query="LOT-2024-001 HOLD 원인",
    )
    state["extracted_info"] = {"lot_id": "LOT-2024-001", "user_id": "user001"}

    result = await log_investigator(state)

    assert "log_entries" in result
    assert len(result["log_entries"]) > 0
    assert result["is_program_issue"] == True


@pytest.mark.asyncio
async def test_routing_after_analysis():
    """조건부 엣지: 분석 후 라우팅 테스트"""
    from agent.edges.conditions import route_after_analysis

    state = create_initial_state("u", "s", "q")
    state["missing_info"] = []
    assert route_after_analysis(state) == "mes_query_executor"

    state["missing_info"] = ["Lot 번호가 필요합니다"]
    state["clarification_count"] = 0
    assert route_after_analysis(state) == "clarify_user"

    state["clarification_count"] = 2
    assert route_after_analysis(state) == "mes_query_executor"


@pytest.mark.asyncio
async def test_routing_after_log():
    """조건부 엣지: 로그 조회 후 라우팅 테스트"""
    from agent.edges.conditions import route_after_log

    state = create_initial_state("u", "s", "q")
    state["is_program_issue"] = False
    state["related_programs"] = []
    assert route_after_log(state) == "rag_retriever"

    state["is_program_issue"] = True
    state["related_programs"] = [{"class_name": "WireBondService"}]
    assert route_after_log(state) == "source_analyzer"


@pytest.mark.asyncio
async def test_rag_retriever():
    """RAG 검색 테스트 (샘플 지식베이스)"""
    from agent.nodes.rag_retriever import rag_retriever

    state = create_initial_state("u", "s", "와이어본딩 Lot HOLD 와이어 장력")
    state["query_category"] = "DATA_ERROR"
    state["error_codes"] = ["MES-ERR-001"]

    result = await rag_retriever(state)

    assert "rag_context" in result
    assert len(result["rag_context"]) > 0
    assert len(result["processing_steps"]) > 0


if __name__ == "__main__":
    asyncio.run(test_initial_state_creation())
    asyncio.run(test_mes_query_executor_mock())
    asyncio.run(test_log_investigator_mock())
    asyncio.run(test_rag_retriever())
    print("\n✅ 모든 테스트 통과")
