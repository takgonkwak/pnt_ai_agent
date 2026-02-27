"""
FastMCP 서버/클라이언트 유틸리티 및 브릿지
- MCP 서버 도구 호출 기능
- MCP 도구의 LangChain 도구 변환 기능
- Mock 데이터 로드 기능
"""
import json
import os
import logging
from typing import Any, List
from fastmcp import Client, FastMCP
from langchain_core.tools import StructuredTool

logger = logging.getLogger(__name__)

MOCK_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "mock_data.json")

def load_mock_data():
    """mock_data.json 파일에서 데이터를 로드합니다."""
    try:
        if os.path.exists(MOCK_DATA_PATH):
            with open(MOCK_DATA_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            logger.warning(f"Mock data file not found at {MOCK_DATA_PATH}")
            return {}
    except Exception as e:
        logger.error(f"Error loading mock data: {e}")
        return {}


async def call_mcp(server: FastMCP, tool: str, **kwargs) -> Any:
    """
    FastMCP 서버 도구 호출 후 결과 역직렬화

    Returns:
        dict/list/str: 도구 반환값 (JSON 자동 파싱)
        None: 결과 없음
    """
    async with Client(server) as client:
        result = await client.call_tool(tool, kwargs)

    if result is None or result.is_error:
        return None

    # fastmcp 2+ : CallToolResult.data 에 이미 역직렬화된 값이 들어 있음
    return result.data


def get_langchain_tools(mcp_server: FastMCP) -> List[StructuredTool]:
    """
    FastMCP 서버에 등록된 도구들을 LangChain용 StructuredTool 목록으로 변환합니다.
    """
    langchain_tools = []
    
    # FastMCP 내부에 등록된 도구 목록 가져오기 (버전 호환성 고려)
    # _local_provider.tools 를 우선 시도
    tools_dict = {}
    if hasattr(mcp_server, "_local_provider") and hasattr(mcp_server._local_provider, "tools"):
        tools_dict = mcp_server._local_provider.tools
    elif hasattr(mcp_server, "_tools"):
        tools_dict = mcp_server._tools
        
    for name, tool in tools_dict.items():
        try:
            # MCP 도구를 LangChain StructuredTool로 매핑
            # tool.fn 은 실제 실행될 함수
            st = StructuredTool.from_function(
                func=tool.fn,
                name=name,
                description=tool.description,
            )
            langchain_tools.append(st)
            logger.info(f"Converted MCP tool to LangChain: {name}")
        except Exception as e:
            logger.error(f"Failed to convert tool {name}: {e}")
            
    return langchain_tools
