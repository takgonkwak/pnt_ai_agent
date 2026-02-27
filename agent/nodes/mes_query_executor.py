"""
Node 2: mes_query_executor
역할: MES DB에서 사용자 문의 관련 현황 데이터 조회
      FastMCP mes-db-server를 통해 동적 쿼리 실행
"""
import logging
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from agent.state import MESAgentState
from config import settings
from servers.mes_db_server import mes_db_mcp
from servers.mcp_utils import call_mcp, get_langchain_tools

logger = logging.getLogger(__name__)


async def mes_query_executor(state: MESAgentState) -> MESAgentState:
    """MES DB 조회 노드 (LLM Tool Calling 기반)"""
    logger.info(f"[mes_query_executor] session={state['session_id']}")

    query_results = {}
    executed_sqls = []

    # 1. MCP 도구를 LangChain 도구로 변환
    tools = get_langchain_tools(mes_db_mcp)
    
    # 2. LLM 초기화 및 도구 바인딩
    llm = ChatOpenAI(
        model=settings.llm_model,
        max_tokens=1024,
        api_key=settings.openai_api_key,
    ).bind_tools(tools)

    try:
        # 3. LLM에게 사용자 질의와 추출 정보를 전달하여 도구 선택 요청
        user_query = state.get("original_query", "")
        intent = state.get("query_intent", "")
        extracted = state.get("extracted_info", {})
        
        prompt = f"""사용자의 MES 관련 문의를 해결하기 위해 필요한 DB 조회를 수행하세요.
문의내용: {user_query}
의도분석: {intent}
추출정보: {extracted}

질의에 가장 적합한 도구를 선택하고 필요한 파라미터를 채워 호출하세요.
만약 적합한 특정 조회 도구가 없다면 'run_sql'을 사용하여 직접 SELECT 쿼리를 작성하세요.
"""
        
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        
        # 4. LLM이 결정한 도구 실행 (Tool Calls)
        if response.tool_calls:
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                args = tool_call["args"]
                
                logger.info(f"[mes_query_executor] Calling tool: {tool_name} with args: {args}")
                
                # 실제 MCP 도구 호출
                result = await call_mcp(mes_db_mcp, tool_name, **args)
                
                # 결과 저장
                result_key = f"{tool_name}_result"
                query_results[result_key] = result
                executed_sqls.append(f"{tool_name.upper()}: {args}")
        else:
            logger.warning("[mes_query_executor] LLM이 호출할 도구를 찾지 못했습니다.")
            query_results["message"] = "질의에 적합한 조회 도구를 찾지 못해 현황 조회를 수행하지 못했습니다."

    except Exception as e:
        logger.error(f"[mes_query_executor] 오류: {e}")
        query_results["error"] = str(e)

    return {
        **state,
        "mes_query_results": query_results,
        "mes_query_sqls": executed_sqls,
        "processing_steps": state.get("processing_steps", []) + [
            f"✅ LLM 기반 MES 도구 호출 완료: {list(query_results.keys())}"
        ],
    }
