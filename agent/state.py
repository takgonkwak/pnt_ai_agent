from typing import TypedDict, Optional, List, Dict, Any
from models.schemas import QueryCategory, EscalationLevel


class MESAgentState(TypedDict, total=False):
    # ── 입력 ──────────────────────────────────────────────
    user_id: str
    session_id: str
    original_query: str                 # 원본 질의 텍스트
    screenshots: List[str]              # 스크린샷 base64 목록
    screenshot_analysis: str            # Vision LLM 이미지 분석 결과

    # ── 질의 분석 결과 ────────────────────────────────────
    query_intent: str                   # 질의 의도 (1문장 요약)
    query_category: str                 # QueryCategory 값
    extracted_info: Dict[str, Any]      # lot_id, equip_id, timestamp 등
    missing_info: List[str]             # 추가 필요 정보 목록
    clarification_question: str         # 사용자에게 물어볼 질문
    clarification_count: int            # 재문의 횟수 (최대 MAX_CLARIFICATION_TURNS)

    # ── MES 조회 결과 ─────────────────────────────────────
    mes_query_results: Dict[str, Any]   # DB 조회 결과 구조체
    mes_query_sqls: List[str]           # 실행된 SQL 목록 (감사용)

    # ── 로그 조회 결과 ────────────────────────────────────
    log_entries: List[Dict[str, Any]]   # 관련 로그 목록
    error_codes: List[str]              # 발견된 에러 코드
    related_programs: List[Dict[str, Any]]  # 관련 프로그램 (class, method, line)
    is_program_issue: bool              # 프로그램 문제 여부 판단

    # ── 소스 분석 결과 (조건부) ───────────────────────────
    source_contents: Dict[str, str]     # {class_name: source_code}
    source_analysis: Dict[str, Any]     # 분석 결과 (원인, 조건, 영향범위 등)
    root_cause: str                     # 근본 원인 요약

    # ── RAG 검색 결과 ─────────────────────────────────────
    rag_context: List[Dict[str, Any]]   # 검색된 관련 문서 청크
    similar_cases: List[Dict[str, Any]] # 유사 장애 사례

    # ── 최종 답변 ─────────────────────────────────────────
    final_answer: str                   # 생성된 답변 (마크다운)
    answer_confidence: float            # 답변 신뢰도 0.0 ~ 1.0
    escalation_required: bool           # 에스컬레이션 필요 여부
    escalation_level: str               # EscalationLevel 값

    # ── 메타 ──────────────────────────────────────────────
    processing_steps: List[str]         # 처리 단계 이력 (디버그/UI용)
    error_state: Optional[str]          # 내부 오류 발생 시 메시지


def create_initial_state(
    user_id: str,
    session_id: str,
    query: str,
    screenshots: Optional[List[str]] = None,
) -> MESAgentState:
    """초기 State 생성 헬퍼"""
    return MESAgentState(
        user_id=user_id,
        session_id=session_id,
        original_query=query,
        screenshots=screenshots or [],
        screenshot_analysis="",
        query_intent="",
        query_category=QueryCategory.UNKNOWN,
        extracted_info={},
        missing_info=[],
        clarification_question="",
        clarification_count=0,
        mes_query_results={},
        mes_query_sqls=[],
        log_entries=[],
        error_codes=[],
        related_programs=[],
        is_program_issue=False,
        source_contents={},
        source_analysis={},
        root_cause="",
        rag_context=[],
        similar_cases=[],
        final_answer="",
        answer_confidence=0.0,
        escalation_required=False,
        escalation_level=EscalationLevel.NONE,
        processing_steps=[],
        error_state=None,
    )
