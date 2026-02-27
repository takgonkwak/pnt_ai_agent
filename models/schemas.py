from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class QueryCategory(str, Enum):
    DATA_ERROR = "DATA_ERROR"           # 데이터 오류/불일치
    PERFORMANCE = "PERFORMANCE"         # 성능 문제 (느림/타임아웃)
    PROGRAM_BUG = "PROGRAM_BUG"         # 기능 오동작/버그
    PROCESS_QUESTION = "PROCESS_QUESTION"  # 업무 프로세스 문의
    PERMISSION = "PERMISSION"           # 권한/접근 문제
    UNKNOWN = "UNKNOWN"                 # 분류 불가


class LogLevel(str, Enum):
    ERROR = "ERROR"
    WARN = "WARN"
    INFO = "INFO"
    DEBUG = "DEBUG"


class EscalationLevel(str, Enum):
    NONE = "NONE"
    L1 = "L1"   # 일반 담당자
    L2 = "L2"   # 시스템 개발자
    L3 = "L3"   # DBA / 인프라


# ────────────────────────────────────────────────────────
# 채팅 API 요청/응답
# ────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str = Field(..., description="세션 ID")
    user_id: str = Field(..., description="사용자 ID")
    message: str = Field(..., description="질의 텍스트")
    screenshots: List[str] = Field(
        default_factory=list,
        description="스크린샷 base64 인코딩 목록"
    )


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    confidence: float
    escalation_required: bool
    escalation_level: EscalationLevel = EscalationLevel.NONE
    processing_steps: List[str]
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    # Human-in-the-Loop: 재문의 필요 시 True, 같은 session_id로 재호출하면 그래프 재개
    needs_clarification: bool = False
    clarification_question: str = ""


# ────────────────────────────────────────────────────────
# MES 조회 결과
# ────────────────────────────────────────────────────────

class LotInfo(BaseModel):
    lot_id: str
    product_id: Optional[str] = None
    process_id: Optional[str] = None
    equip_id: Optional[str] = None
    status: Optional[str] = None
    qty: Optional[int] = None
    hold_reason: Optional[str] = None
    last_updated: Optional[str] = None


class EquipStatus(BaseModel):
    equip_id: str
    equip_name: Optional[str] = None
    status: Optional[str] = None
    current_recipe: Optional[str] = None
    alarm_code: Optional[str] = None


# ────────────────────────────────────────────────────────
# 로그 관련
# ────────────────────────────────────────────────────────

class LogEntry(BaseModel):
    timestamp: str
    level: LogLevel
    program: Optional[str] = None
    class_name: Optional[str] = None
    method_name: Optional[str] = None
    line_number: Optional[int] = None
    message: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    stack_trace: Optional[str] = None


class ProgramReference(BaseModel):
    program_id: str
    class_name: str
    method_name: Optional[str] = None
    line_number: Optional[int] = None
    error_message: str
    stack_trace: Optional[str] = None


# ────────────────────────────────────────────────────────
# RAG 검색 결과
# ────────────────────────────────────────────────────────

class RAGDocument(BaseModel):
    doc_id: str
    source: str        # 문서 출처 (파일명, URL 등)
    content: str       # 청크 내용
    category: str      # manual | incident | technical
    score: float       # 유사도 점수
    metadata: Dict[str, Any] = Field(default_factory=dict)
