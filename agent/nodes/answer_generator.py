"""
Node 5: answer_generator
역할: 수집된 모든 컨텍스트(MES 데이터, 로그, 소스분석, RAG)를 통합하여
      구조화된 최종 답변을 생성하고 신뢰도 및 에스컬레이션 여부를 결정
"""
import json
import logging
from typing import List, Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from agent.state import MESAgentState
from models.schemas import EscalationLevel
from config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """당신은 반도체 후공정 MES LT 시스템 전문 기술 지원 담당자입니다.
수집된 모든 데이터를 바탕으로 사용자에게 명확하고 실용적인 답변을 제공하세요.

답변 원칙:
1. 근거 데이터를 명시하세요 (어떤 로그, 어떤 DB 조회 결과 기반)
2. 기술 용어는 사용자 수준에 맞게 조정하세요
3. 불확실한 내용은 명확히 표시하세요 ("확인이 필요합니다", "가능성이 있습니다" 등)
4. 단계별 조치 방법을 명확하게 제시하세요
5. 마크다운 형식으로 작성하세요
"""

ANSWER_PROMPT_TEMPLATE = """다음 정보를 종합하여 MES 시스템 문의에 대한 답변을 작성하세요.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[사용자 원본 질의]
{original_query}

[분석된 의도]
{query_intent}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[MES 현황 조회 결과]
{mes_results}

[시스템 로그 분석]
- 에러코드: {error_codes}
- 주요 에러 로그:
{log_summary}

[소스코드 분석 결과]
{source_analysis}

[관련 문서 및 유사 사례]
{rag_context}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

다음 형식의 JSON으로 응답하세요:
{{
    "answer": "## 현황 요약\n...\n## 원인 분석\n...\n## 조치 방법\n...\n## 참고 사항\n...",
    "confidence": 0.0,
    "escalation_required": false,
    "escalation_level": "NONE|L1|L2|L3",
    "escalation_reason": "에스컬레이션 사유 (required=false면 빈 문자열)"
}}

에스컬레이션 기준:
- L1 (일반 담당자): 업무 프로세스 문의, 권한 요청
- L2 (시스템 개발자): 프로그램 버그 확인, 배치 문제
- L3 (DBA/인프라): DB 데이터 오류, 성능 문제, 인프라 이슈
"""


def _fmt_mes(results: Dict[str, Any]) -> str:
    return json.dumps(results, ensure_ascii=False, indent=2)[:1500] if results else "조회된 MES 데이터 없음"


def _fmt_logs(entries: List[Dict]) -> str:
    if not entries:
        return "관련 로그 없음"
    lines = [f"  [{e.get('level')}] {e.get('timestamp','')} - {e.get('message','')[:100]}"
             for e in entries[:5]]
    if len(entries) > 5:
        lines.append(f"  ... 외 {len(entries)-5}건")
    return "\n".join(lines)


def _fmt_source(analysis: Dict[str, Any]) -> str:
    if not analysis or analysis.get("message"):
        return analysis.get("message", "소스 분석 없음")
    p = analysis.get("primary", {})
    if not p:
        return "소스 분석 결과 없음"
    return (f"- 버그 위치: {p.get('bug_location','N/A')}\n"
            f"- 근본 원인: {p.get('root_cause','N/A')}\n"
            f"- 발생 조건: {p.get('trigger_condition','N/A')}\n"
            f"- 임시 조치: {p.get('workaround','N/A')}\n"
            f"- 심각도: {p.get('severity','N/A')}")


def _fmt_rag(rag_docs: List[Dict], similar_cases: List[Dict]) -> str:
    parts = []
    if rag_docs:
        parts.append("관련 문서:")
        for d in rag_docs[:3]:
            parts.append(f"  [{d.get('source','')}] {d.get('content','')[:150]}")
    if similar_cases:
        parts.append("유사 장애 사례:")
        for c in similar_cases[:2]:
            parts.append(f"  {c.get('content','')[:150]}")
    return "\n".join(parts) if parts else "관련 문서 없음"


def _parse(raw: str) -> Dict[str, Any]:
    text = raw.strip()
    if "```" in text:
        lines = text.split("\n")
        s = next((i for i, l in enumerate(lines) if "```" in l), 0)
        e = next((i for i, l in enumerate(lines) if "```" in l and i > s), len(lines))
        text = "\n".join(lines[s + 1:e])
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"answer": raw, "confidence": 0.5,
                "escalation_required": False, "escalation_level": "NONE", "escalation_reason": ""}


async def answer_generator(state: MESAgentState) -> MESAgentState:
    """최종 답변 생성 노드"""
    logger.info(f"[answer_generator] session={state['session_id']}")

    llm = ChatOpenAI(
        model=settings.llm_model,
        max_tokens=settings.llm_max_tokens,
        api_key=settings.openai_api_key,
    )

    prompt = ANSWER_PROMPT_TEMPLATE.format(
        original_query=state.get("original_query", ""),
        query_intent=state.get("query_intent", ""),
        mes_results=_fmt_mes(state.get("mes_query_results", {})),
        error_codes=", ".join(state.get("error_codes", [])) or "없음",
        log_summary=_fmt_logs(state.get("log_entries", [])),
        source_analysis=_fmt_source(state.get("source_analysis", {})),
        rag_context=_fmt_rag(state.get("rag_context", []), state.get("similar_cases", [])),
    )

    try:
        response = await llm.ainvoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])
        result = _parse(response.content)

        confidence = float(result.get("confidence", 0.5))
        escalation_required = result.get("escalation_required", False)

        if confidence < settings.answer_confidence_threshold:
            escalation_required = True
            result["escalation_level"] = "L1"
            result["escalation_reason"] = (
                f"답변 신뢰도({confidence:.1%})가 임계값({settings.answer_confidence_threshold:.1%}) 미만"
            )

        logger.info(f"[answer_generator] confidence={confidence:.2f}, escalation={escalation_required}")

        return {
            **state,
            "final_answer": result.get("answer", ""),
            "answer_confidence": confidence,
            "escalation_required": escalation_required,
            "escalation_level": result.get("escalation_level", EscalationLevel.NONE),
            "processing_steps": state.get("processing_steps", []) + [
                f"✅ 답변 생성 완료: 신뢰도={confidence:.1%}, 에스컬레이션={escalation_required}"
            ],
        }

    except Exception as e:
        logger.error(f"[answer_generator] 오류: {e}")
        return {
            **state,
            "final_answer": "죄송합니다. 답변 생성 중 오류가 발생했습니다. 담당자에게 문의하세요.",
            "answer_confidence": 0.0,
            "escalation_required": True,
            "escalation_level": EscalationLevel.L1,
            "processing_steps": state.get("processing_steps", []) + [f"❌ 답변 생성 실패: {str(e)}"],
        }
