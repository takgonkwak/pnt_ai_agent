"""
Node 1: query_analyzer
역할: 사용자 질의(텍스트 + 스크린샷) 분석, 의도 분류, 핵심 엔티티 추출,
      추가 정보 필요 여부 판단
"""
import json
import logging
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from agent.state import MESAgentState
from config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """당신은 반도체 후공정 MES(Manufacturing Execution System) IT 시스템 전문가입니다.
사용자의 문의를 정확히 분석하여 다음을 수행하세요:
1. 질의 의도 파악
2. 카테고리 분류
3. 핵심 엔티티 추출 (Lot번호, 설비ID, 공정, 에러메시지, 발생시간 등)
4. 답변에 필요하지만 없는 정보 파악

반드시 JSON 형식으로만 응답하세요.
"""

ANALYSIS_PROMPT_TEMPLATE = """다음 MES 시스템 문의를 분석하세요.

[사용자 질의]
{query}

{screenshot_section}

다음 JSON 구조로 응답하세요 (설명 없이 JSON만):
{{
    "query_intent": "질의 의도 요약 (1~2문장)",
    "query_category": "DATA_ERROR|PERFORMANCE|PROGRAM_BUG|PROCESS_QUESTION|PERMISSION|UNKNOWN",
    "extracted_info": {{
        "lot_id": "Lot번호 또는 null",
        "equip_id": "설비ID 또는 null",
        "process_id": "공정ID 또는 null",
        "product_id": "제품ID 또는 null",
        "user_id": "사용자ID 또는 null",
        "screen_name": "화면명 또는 null",
        "error_message": "에러메시지 또는 null",
        "error_code": "에러코드 또는 null",
        "timestamp": "발생시간 (YYYY-MM-DD HH:MM:SS) 또는 null",
        "additional": {{}}
    }},
    "missing_info": ["부족한 정보1", "부족한 정보2"],
    "clarification_question": "사용자에게 물어볼 질문 (missing_info가 없으면 빈 문자열)",
    "screenshot_findings": "스크린샷에서 발견된 주요 정보 (없으면 빈 문자열)"
}}
"""


def _build_messages(state: MESAgentState) -> list:
    """LLM 메시지 구성 (멀티모달 지원 - OpenAI Vision)"""
    content = []

    # 스크린샷 처리 (OpenAI Vision 형식)
    screenshot_section = ""
    for i, screenshot_b64 in enumerate(state.get("screenshots", [])):
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{screenshot_b64}",
                "detail": "high",
            },
        })
        screenshot_section = (
            f"\n[스크린샷 {i+1}개 첨부됨 - 위 이미지를 분석하여 "
            f"에러 메시지, 화면명, 데이터 등을 추출하세요]"
        )

    prompt_text = ANALYSIS_PROMPT_TEMPLATE.format(
        query=state["original_query"],
        screenshot_section=screenshot_section if screenshot_section else "[스크린샷 없음]",
    )
    content.append({"type": "text", "text": prompt_text})

    return [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=content),
    ]


def _parse_llm_response(raw: str) -> dict:
    """LLM 응답에서 JSON 파싱 (마크다운 코드블록 제거)"""
    text = raw.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return json.loads(text.strip())


def _extract_missing_info(category: str, extracted: dict) -> list:
    """카테고리별 필수 엔티티 누락 여부 검사 (Validator)"""
    missing = []
    
    # 룰셋: 카테고리별 필수 항목 정의 (필요시 도메인에 맞게 커스터마이징)
    required_fields = {
        "DATA_ERROR": ["lot_id", "equip_id"],
        "PERFORMANCE": ["screen_name", "timestamp"],
        "PROGRAM_BUG": ["screen_name", "error_message"],
        "PROCESS_QUESTION": ["process_id"],
        "PERMISSION": ["user_id", "screen_name"]
    }
    
    # 카테고리에 해당하는 필수 항목 확인
    fields_to_check = required_fields.get(category, [])
    for field in fields_to_check:
        val = extracted.get(field)
        if not val or str(val).lower() == "null" or str(val).strip() == "":
            missing.append(field)
            
    return missing


async def query_analyzer(state: MESAgentState) -> MESAgentState:
    """질의 분석 노드"""
    logger.info(f"[query_analyzer] session={state['session_id']} query='{state['original_query'][:50]}...'")

    llm = ChatOpenAI(
        model=settings.llm_model,
        max_tokens=1024,
        api_key=settings.openai_api_key,
    )

    try:
        messages = _build_messages(state)
        response = await llm.ainvoke(messages)
        result = _parse_llm_response(response.content)

        category = result.get("query_category", "UNKNOWN")
        extracted = result.get("extracted_info", {})
        missing_info = result.get("missing_info", [])
        
        if not isinstance(missing_info, list):
            missing_info = [missing_info] if missing_info else []

        # 파이썬 로직으로 누락된 정보(Validator) 추출 및 병합
        code_missing = _extract_missing_info(category, extracted)
        for field in code_missing:
            if field not in missing_info:
                missing_info.append(field)

        clarification_question = result.get("clarification_question", "")
        # 추가 질문이 비어있는데 missing_info가 있다면 기본 안내 메시지 생성
        if missing_info and not clarification_question:
            clarification_question = f"문의 내용을 정확히 파악하기 위해 다음 정보가 추가로 필요합니다: {', '.join(missing_info)}"

        updates: dict[str, Any] = {
            "query_intent": result.get("query_intent", ""),
            "query_category": category,
            "extracted_info": extracted,
            "missing_info": missing_info,
            "clarification_question": clarification_question,
            "screenshot_analysis": result.get("screenshot_findings", ""),
            "processing_steps": state.get("processing_steps", []) + [
                f"✅ 질의 분석 완료: 카테고리={category}"
            ],
        }

        logger.info(f"[query_analyzer] category={updates['query_category']}, missing={updates['missing_info']}")
        return {**state, **updates}

    except Exception as e:
        logger.error(f"[query_analyzer] 오류: {e}")
        return {
            **state,
            "error_state": f"질의 분석 실패: {str(e)}",
            "processing_steps": state.get("processing_steps", []) + ["❌ 질의 분석 실패"],
        }
