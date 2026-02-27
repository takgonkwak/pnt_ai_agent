"""
Node 4: source_analyzer (조건부)
역할: 로그에서 식별된 프로그램의 소스코드를 조회하고 LLM으로 버그 원인 분석
      is_program_issue=True 일 때만 실행됨
"""
import json
import logging
from typing import Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from agent.state import MESAgentState
from config import settings
from servers.source_server import source_mcp
from servers.mcp_utils import call_mcp

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """당신은 반도체 MES 시스템 소프트웨어 개발 전문가입니다.
제공된 소스코드와 에러 로그를 분석하여 버그의 근본 원인을 도출하세요.
기술적으로 정확하고 실용적인 분석을 제공하세요.
"""

ANALYSIS_PROMPT_TEMPLATE = """다음 소스코드와 에러 로그를 분석하여 버그 원인을 파악하세요.

[에러 메시지]
{error_message}

[에러 코드]
{error_codes}

[스택 트레이스]
{stack_trace}

[소스코드: {class_name}.{method_name} (line {line_number})]
```java
{source_code}
```

[관련 MES 데이터 현황]
{mes_context}

다음 JSON 구조로 분석 결과를 응답하세요:
{{
    "bug_location": "버그가 있는 정확한 위치 (클래스.메서드:라인번호)",
    "root_cause": "근본 원인 설명 (기술적으로 정확하게)",
    "trigger_condition": "어떤 조건에서 이 버그가 발생하는가",
    "impact_scope": "영향 범위 (어떤 기능/데이터에 영향)",
    "workaround": "임시 해결 방법 (사용자가 즉시 적용 가능한 방법)",
    "fix_direction": "근본 수정 방향 (개발팀에게 전달할 내용)",
    "severity": "CRITICAL|HIGH|MEDIUM|LOW",
    "confidence": 0.0
}}
"""


async def source_analyzer(state: MESAgentState) -> MESAgentState:
    """소스코드 분석 노드"""
    logger.info(f"[source_analyzer] session={state['session_id']}")

    llm = ChatOpenAI(
        model=settings.llm_model,
        max_tokens=2048,
        api_key=settings.openai_api_key,
    )

    related_programs = state.get("related_programs", [])
    if not related_programs:
        return {
            **state,
            "source_analysis": {"message": "분석할 프로그램 정보 없음"},
            "root_cause": "",
            "processing_steps": state.get("processing_steps", []) + [
                "⚠️ 소스분석 건너뜀: 관련 프로그램 없음"
            ],
        }

    source_contents: Dict[str, str] = {}
    analysis_results = []

    for prog in related_programs[:3]:
        class_name = prog.get("class_name", "")
        method_name = prog.get("method_name", "")
        line_number = prog.get("line_number", 0)

        try:
            source_code = await call_mcp(
                source_mcp, "get_source_by_class",
                class_name=class_name,
                method_name=method_name,
                line_number=line_number,
            )

            if not source_code:
                logger.warning(f"[source_analyzer] 소스 없음: {class_name}")
                continue

            source_contents[class_name] = source_code

            prompt = ANALYSIS_PROMPT_TEMPLATE.format(
                error_message=prog.get("error_message", ""),
                error_codes=", ".join(state.get("error_codes", [])),
                stack_trace=prog.get("stack_trace", "")[:500],
                class_name=class_name,
                method_name=method_name,
                line_number=line_number,
                source_code=str(source_code)[:3000],
                mes_context=json.dumps(state.get("mes_query_results", {}),
                                       ensure_ascii=False)[:500],
            )

            response = await llm.ainvoke([
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ])

            analysis = _parse_json(response.content)
            analysis["class_name"] = class_name
            analysis_results.append(analysis)

            logger.info(f"[source_analyzer] 분석 완료: {class_name}, "
                        f"severity={analysis.get('severity')}")

        except Exception as e:
            logger.error(f"[source_analyzer] {class_name} 분석 오류: {e}")

    if not analysis_results:
        root_cause = "소스코드를 조회할 수 없어 직접 분석이 불가합니다."
        combined: Dict[str, Any] = {"message": root_cause}
    else:
        best = max(analysis_results, key=lambda x: (
            {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}.get(x.get("severity", "LOW"), 0)
            + float(x.get("confidence", 0))
        ))
        root_cause = best.get("root_cause", "")
        combined = {"analyses": analysis_results, "primary": best}

    return {
        **state,
        "source_contents": source_contents,
        "source_analysis": combined,
        "root_cause": root_cause,
        "processing_steps": state.get("processing_steps", []) + [
            f"✅ 소스 분석 완료: {len(analysis_results)}개 프로그램 분석"
        ],
    }


def _parse_json(raw: str) -> dict:
    text = raw.strip()
    if "```" in text:
        lines = text.split("\n")
        start = next((i for i, l in enumerate(lines) if "```" in l), 0)
        end = next((i for i, l in enumerate(lines) if "```" in l and i > start), len(lines))
        text = "\n".join(lines[start + 1:end])
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw_analysis": raw, "confidence": 0.3}
