# MES LT 시스템 사용자 문의 응대 AI Agent 설계 문서

## 1. 개요

### 1.1 시스템 목표
반도체 후공정 MES(Manufacturing Execution System) LT(Lot Tracking) 시스템에서 발생하는 사용자 문의를 자동으로 접수, 분석, 조사, 답변하는 AI Agent 시스템 구축

### 1.2 핵심 기술 스택
| 기술 | 역할 |
|------|------|
| **LangGraph** | Agent 워크플로우 오케스트레이션 (State Machine 기반 다단계 처리) |
| **MCP (Model Context Protocol)** | MES DB, 로그 시스템, 소스코드 저장소 등 외부 도구 표준 연동 |
| **RAG (Retrieval-Augmented Generation)** | MES 매뉴얼, 과거 장애 이력, 업무 지식 기반 답변 생성 |
| **LLM (Claude / GPT-4o)** | 자연어 이해, 이미지 분석(스크린샷), 답변 생성 |

---

## 2. 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────────────┐
│                         사용자 인터페이스                              │
│              (웹 채팅 / 슬랙 / 이메일 / MES 내 팝업)                   │
│                  [텍스트 입력] + [스크린샷 첨부]                        │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      LangGraph Agent Engine                          │
│                                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐           │
│  │ Node 1   │→ │ Node 2   │→ │ Node 3   │→ │ Node 4   │→ Node 5  │
│  │질의분석   │  │MES현황   │  │로그조회   │  │소스분석   │  │답변생성 │
│  │& 보완    │  │조회      │  │          │  │(조건부)   │          │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘           │
│       │              │              │              │                  │
│       ▼              ▼              ▼              ▼                  │
│  ┌──────────────────────────────────────────────────┐               │
│  │                 Agent State Store                 │               │
│  │  (질의, 수집정보, MES데이터, 로그, 소스분석결과)     │               │
│  └──────────────────────────────────────────────────┘               │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
┌─────────────────┐  ┌──────────────┐  ┌──────────────────┐
│   MCP Servers   │  │  RAG Engine  │  │  Vector DB       │
│                 │  │              │  │                  │
│ • MES DB Server │  │ • 문서 검색  │  │ • MES 매뉴얼     │
│ • Log Server    │  │ • 유사사례   │  │ • 장애 이력      │
│ • Source Server │  │   검색       │  │ • 업무 지식      │
│ • Screen Server │  └──────────────┘  │ • 에러 코드 DB   │
└─────────────────┘                    └──────────────────┘
```

---

## 3. LangGraph 워크플로우 상세 설계

### 3.1 State 정의

```python
from typing import TypedDict, Optional, List, Annotated
from langgraph.graph import StateGraph

class MESAgentState(TypedDict):
    # 입력
    user_id: str
    session_id: str
    original_query: str                   # 원본 질의 텍스트
    screenshots: List[str]                # 스크린샷 파일 경로 or base64

    # 질의 분석 결과
    query_intent: str                     # 질의 의도 분류
    query_category: str                   # 카테고리 (데이터오류/성능/UI버그/프로세스등)
    extracted_info: dict                  # 추출된 정보 (Lot번호, 설비, 시간대 등)
    missing_info: List[str]               # 추가 필요 정보 목록
    clarification_count: int              # 사용자 재문의 횟수

    # MES 조회 결과
    mes_query_results: dict               # DB/화면 조회 결과
    mes_query_sql: List[str]              # 실행된 SQL 목록

    # 로그 조회 결과
    log_entries: List[dict]               # 관련 로그 목록
    error_codes: List[str]                # 발견된 에러 코드
    related_programs: List[str]           # 관련 프로그램 목록

    # 소스 분석 결과
    is_program_issue: bool                # 프로그램 문제 여부
    source_analysis: dict                 # 소스 분석 결과
    root_cause: str                       # 근본 원인

    # RAG 검색 결과
    rag_context: List[dict]               # 검색된 관련 문서
    similar_cases: List[dict]             # 유사 장애 사례

    # 최종 답변
    final_answer: str                     # 생성된 답변
    answer_confidence: float              # 답변 신뢰도 (0~1)
    escalation_required: bool             # 담당자 에스컬레이션 필요 여부

    # 메타
    processing_steps: List[str]           # 처리 단계 이력
    error_state: Optional[str]            # 오류 상태
```

### 3.2 노드 흐름도

```
[START]
   │
   ▼
┌─────────────────────────────────────────┐
│  Node: query_analyzer                   │
│  - 텍스트 + 스크린샷 멀티모달 분석         │
│  - 의도 분류, 엔티티 추출                  │
│  - 부족 정보 파악                         │
└──────────┬──────────────────────────────┘
           │
    ┌──────┴──────┐
    │ 추가정보    │
    │ 필요?       │
    └──┬──────┬──┘
    YES│      │NO
       ▼      ▼
┌──────────┐  │
│ Node:    │  │
│clarify_  │  │
│user      │  │
│(사용자   │  │
│ 재문의)  │  │
└─────┬────┘  │
      │       │
      └───────┘
           │
           ▼
┌─────────────────────────────────────────┐
│  Node: mes_query_executor               │
│  - MCP를 통해 MES DB 조회               │
│  - 동적 SQL 생성 및 실행                 │
│  - MES 화면 데이터 스크래핑 (필요시)     │
└──────────┬──────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────┐
│  Node: log_investigator                 │
│  - MCP를 통해 시스템 로그 검색           │
│  - 에러 로그 필터링 및 분류              │
│  - 관련 프로그램 식별                    │
└──────────┬──────────────────────────────┘
           │
    ┌──────┴──────┐
    │ 프로그램     │
    │ 문제?        │
    └──┬──────┬──┘
    YES│      │NO
       ▼      │
┌──────────┐  │
│ Node:    │  │
│source_   │  │
│analyzer  │  │
│- 소스코드│  │
│  조회    │  │
│- 버그분석│  │
└─────┬────┘  │
      │       │
      └───────┘
           │
           ▼
┌─────────────────────────────────────────┐
│  Node: rag_retriever                    │
│  - 유사 사례 벡터 검색                   │
│  - MES 매뉴얼 관련 항목 검색             │
│  - 에러코드 설명 검색                    │
└──────────┬──────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────┐
│  Node: answer_generator                 │
│  - 수집된 모든 컨텍스트 통합              │
│  - 구조화된 답변 생성                    │
│  - 신뢰도 평가 및 에스컬레이션 판단      │
└──────────┬──────────────────────────────┘
           │
    ┌──────┴──────┐
    │ 에스컬레이션 │
    │ 필요?        │
    └──┬──────┬──┘
    YES│      │NO
       ▼      ▼
┌──────────┐ [END]
│ Node:    │
│escalate_ │
│to_human  │
└──────────┘
```

### 3.3 조건부 엣지 (Conditional Edges)

```python
def route_after_analysis(state: MESAgentState) -> str:
    if state["missing_info"] and state["clarification_count"] < 2:
        return "clarify_user"
    return "mes_query_executor"

def route_after_log(state: MESAgentState) -> str:
    if state["is_program_issue"] and state["related_programs"]:
        return "source_analyzer"
    return "rag_retriever"

def route_after_answer(state: MESAgentState) -> str:
    if state["escalation_required"] or state["answer_confidence"] < 0.4:
        return "escalate_to_human"
    return END
```

---

## 4. MCP (Model Context Protocol) 서버 설계

### 4.1 MCP 서버 구성

```
┌──────────────────────────────────────────────────────────────┐
│                    MCP Server Registry                        │
│                                                              │
│  ┌─────────────────┐   ┌─────────────────┐                 │
│  │ mes-db-server   │   │ log-server      │                 │
│  │                 │   │                 │                 │
│  │ Tools:          │   │ Tools:          │                 │
│  │ • query_lot     │   │ • search_logs   │                 │
│  │ • query_wip     │   │ • get_error_log │                 │
│  │ • query_equip   │   │ • get_app_log   │                 │
│  │ • query_recipe  │   │ • search_by_pgm │                 │
│  │ • run_sql       │   │ • get_stack_trace│                │
│  └─────────────────┘   └─────────────────┘                 │
│                                                              │
│  ┌─────────────────┐   ┌─────────────────┐                 │
│  │ source-server   │   │ screen-server   │                 │
│  │                 │   │                 │                 │
│  │ Tools:          │   │ Tools:          │                 │
│  │ • get_source    │   │ • capture_screen│                 │
│  │ • get_class     │   │ • analyze_image │                 │
│  │ • get_method    │   │ • get_ui_data   │                 │
│  │ • search_code   │   │                 │                 │
│  │ • git_blame     │   │                 │                 │
│  └─────────────────┘   └─────────────────┘                 │
│                                                              │
│  ┌─────────────────┐                                        │
│  │ notify-server   │                                        │
│  │                 │                                        │
│  │ Tools:          │                                        │
│  │ • send_slack    │                                        │
│  │ • create_ticket │                                        │
│  │ • escalate      │                                        │
│  └─────────────────┘                                        │
└──────────────────────────────────────────────────────────────┘
```

### 4.2 MCP Tool 상세 명세

#### mes-db-server

```python
# Tool: query_lot
{
    "name": "query_lot",
    "description": "Lot 번호로 MES에서 현재 Lot 상태, 위치, 이력 조회",
    "inputSchema": {
        "lot_id": "string",           # Lot 번호 (필수)
        "include_history": "boolean",  # 이력 포함 여부
        "from_date": "string",         # 조회 시작일 (YYYY-MM-DD)
        "to_date": "string"            # 조회 종료일
    }
}

# Tool: query_wip
{
    "name": "query_wip",
    "description": "WIP(Work In Process) 현황 조회 - 설비, 공정, 제품별",
    "inputSchema": {
        "equip_id": "string",
        "process_id": "string",
        "product_id": "string",
        "status": "string"  # HOLD/RUN/WAIT/COMPLETE
    }
}

# Tool: run_sql
{
    "name": "run_sql",
    "description": "MES DB에 읽기 전용 SELECT 쿼리 실행 (Write 불가)",
    "inputSchema": {
        "sql": "string",
        "max_rows": "integer"  # 최대 1000행
    }
}
```

#### log-server

```python
# Tool: search_logs
{
    "name": "search_logs",
    "description": "키워드, 시간 범위, 로그 레벨로 시스템 로그 검색",
    "inputSchema": {
        "keywords": ["string"],
        "from_time": "string",   # ISO 8601
        "to_time": "string",
        "log_level": "string",   # ERROR/WARN/INFO/DEBUG
        "user_id": "string",
        "program_id": "string",
        "max_results": "integer"
    }
}

# Tool: get_stack_trace
{
    "name": "get_stack_trace",
    "description": "특정 에러 발생 시점의 스택 트레이스 조회",
    "inputSchema": {
        "error_id": "string",
        "session_id": "string",
        "timestamp": "string"
    }
}
```

#### source-server

```python
# Tool: get_source_by_class
{
    "name": "get_source_by_class",
    "description": "로그에 표시된 클래스명으로 소스코드 조회",
    "inputSchema": {
        "class_name": "string",
        "method_name": "string",   # optional
        "line_number": "integer"   # optional
    }
}

# Tool: search_code
{
    "name": "search_code",
    "description": "코드베이스에서 특정 패턴이나 에러 코드 검색",
    "inputSchema": {
        "query": "string",
        "file_pattern": "string",  # e.g., "*.java", "*.py"
        "module": "string"
    }
}
```

---

## 5. RAG 시스템 설계

### 5.1 지식 베이스 구성

```
┌──────────────────────────────────────────────────────────────┐
│                     RAG Knowledge Base                        │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Knowledge Source 1: MES 매뉴얼 & 업무 지식           │   │
│  │  - MES 화면 사용 설명서                               │   │
│  │  - 업무 프로세스 정의서                               │   │
│  │  - 에러 코드 정의 및 처리 방법                        │   │
│  │  - FAQ 문서                                           │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Knowledge Source 2: 과거 장애 이력 (Ticket DB)       │   │
│  │  - 장애 티켓 (증상, 원인, 조치 내용)                  │   │
│  │  - 동일/유사 문제 해결 사례                           │   │
│  │  - 재발 방지 조치 내역                                │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Knowledge Source 3: 기술 문서                        │   │
│  │  - 시스템 아키텍처 문서                               │   │
│  │  - DB 스키마 & 테이블 정의                            │   │
│  │  - API 명세서                                         │   │
│  │  - 배포/운영 가이드                                   │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
                               │
                    ┌──────────┴──────────┐
                    │    Chunking &        │
                    │    Embedding         │
                    │  (chunk_size: 512,   │
                    │   overlap: 64)       │
                    └──────────┬──────────┘
                               │
                    ┌──────────┴──────────┐
                    │    Vector DB         │
                    │  (ChromaDB /         │
                    │   pgvector /         │
                    │   Weaviate)          │
                    └─────────────────────┘
```

### 5.2 Hybrid 검색 전략

```python
class MESRAGRetriever:
    """
    Sparse(BM25) + Dense(Vector) 검색 결합으로 검색 정확도 향상
    """

    def retrieve(self, query: str, context: dict) -> List[Document]:
        # 1. 쿼리 확장: 기술 용어, 동의어 추가
        expanded_query = self.expand_query(query, context)

        # 2. Dense 검색 (의미 기반)
        dense_results = self.vector_store.similarity_search(
            expanded_query, k=10
        )

        # 3. Sparse 검색 (키워드 기반, 에러코드/Lot번호 등 정확 매칭)
        sparse_results = self.bm25.search(
            keywords=context.get("error_codes", []) + [query],
            k=10
        )

        # 4. RRF(Reciprocal Rank Fusion)로 결과 통합
        merged = self.reciprocal_rank_fusion(dense_results, sparse_results)

        # 5. 카테고리 필터링 (질의 카테고리와 문서 카테고리 매칭)
        filtered = self.filter_by_category(merged, context["query_category"])

        return filtered[:5]
```

---

## 6. 노드별 상세 구현 명세

### 6.1 Node 1: query_analyzer

```python
async def query_analyzer(state: MESAgentState) -> MESAgentState:
    """
    역할: 사용자 질의 분석, 멀티모달 처리, 필요 정보 식별

    처리 내용:
    1. 스크린샷이 있으면 Vision LLM으로 화면 내용 추출
       - 에러 메시지, 화면명, Lot번호, 설비ID 등 추출
    2. 텍스트 질의 + 이미지 분석 결과 통합
    3. 의도 분류:
       - DATA_ERROR: 데이터 오류/불일치
       - PERFORMANCE: 화면 느림/타임아웃
       - PROGRAM_BUG: 기능 오동작
       - PROCESS_QUESTION: 업무 프로세스 문의
       - PERMISSION: 권한/접근 문제
    4. 핵심 엔티티 추출:
       - lot_id, equip_id, process_id, user_id
       - error_message, timestamp, screen_name
    5. 부족 정보 목록 생성
    """

    prompt = f"""
    MES LT 시스템 문의를 분석하세요.

    [사용자 질의]
    {state['original_query']}

    [스크린샷 분석 결과]
    {state.get('screenshot_analysis', '없음')}

    다음을 JSON으로 추출하세요:
    - query_intent: 질의 의도 (1문장)
    - query_category: DATA_ERROR|PERFORMANCE|PROGRAM_BUG|PROCESS_QUESTION|PERMISSION
    - extracted_info: {{lot_id, equip_id, process_id, error_message, timestamp, screen_name}}
    - missing_info: 답변을 위해 추가로 필요한 정보 목록
    - clarification_question: 사용자에게 물어볼 질문 (missing_info가 있을 때만)
    """

    # LLM 호출 (멀티모달)
    ...
```

### 6.2 Node 2: mes_query_executor

```python
async def mes_query_executor(state: MESAgentState) -> MESAgentState:
    """
    역할: MES DB 및 화면에서 관련 현황 데이터 조회

    처리 내용:
    1. 질의 카테고리와 추출 엔티티를 기반으로 조회 전략 결정
    2. MCP mes-db-server 도구 호출:
       - Lot 정보, WIP 현황, 설비 상태, 레시피 정보 등
       - LLM이 동적으로 필요한 SQL 생성
    3. 조회 결과를 구조화하여 State에 저장

    보안:
    - SELECT만 허용, DML 차단
    - 최대 1000행 제한
    - 접근 가능 테이블 화이트리스트 관리
    """

    # LLM에게 MCP 도구 제공하여 자율 조회
    tools = [mes_mcp_client.get_tool("query_lot"),
             mes_mcp_client.get_tool("query_wip"),
             mes_mcp_client.get_tool("run_sql")]

    result = await llm_with_tools.ainvoke(
        f"다음 문의에 필요한 MES 현황을 조회하세요: {state['query_intent']}"
        f"엔티티: {state['extracted_info']}",
        tools=tools
    )
    ...
```

### 6.3 Node 3: log_investigator

```python
async def log_investigator(state: MESAgentState) -> MESAgentState:
    """
    역할: 시스템 로그에서 문의 관련 이벤트 및 에러 조사

    처리 내용:
    1. 시간 범위 + 사용자/Lot/설비 기반 로그 검색
    2. 에러 레벨 로그 우선 추출
    3. 스택 트레이스에서 프로그램명, 클래스명, 라인 번호 추출
    4. 관련 프로그램 목록 생성
    5. 프로그램 문제 여부 판단

    로그 우선순위:
    - ERROR > WARN > INFO > DEBUG
    - 사용자 세션 로그 > 배치 로그
    - 발생 시점 ±30분 범위 우선
    """

    log_search_params = {
        "from_time": calculate_search_start(state["extracted_info"].get("timestamp")),
        "to_time": calculate_search_end(state["extracted_info"].get("timestamp")),
        "user_id": state["extracted_info"].get("user_id"),
        "keywords": extract_keywords(state),
        "log_level": "ERROR",
        "max_results": 50
    }

    logs = await log_mcp_client.call_tool("search_logs", log_search_params)
    programs = extract_program_references(logs)

    state["log_entries"] = logs
    state["related_programs"] = programs
    state["is_program_issue"] = len(programs) > 0 and has_error_logs(logs)
    ...
```

### 6.4 Node 4: source_analyzer (조건부)

```python
async def source_analyzer(state: MESAgentState) -> MESAgentState:
    """
    역할: 로그에서 발견된 프로그램 소스코드 분석으로 근본 원인 도출

    처리 내용:
    1. 스택 트레이스에서 클래스/메서드/라인 번호 파싱
    2. MCP source-server로 해당 소스코드 조회
    3. LLM으로 소스코드 + 에러 메시지 분석:
       - 버그 위치 특정
       - 에러 발생 조건 분석
       - 영향 범위 파악
    4. 임시 해결 방법(Workaround) 제안
    5. 수정 방향 제안

    입력: log_entries, related_programs
    출력: source_analysis, root_cause
    """

    for program in state["related_programs"]:
        source = await source_mcp_client.call_tool("get_source_by_class", {
            "class_name": program["class_name"],
            "method_name": program.get("method_name"),
            "line_number": program.get("line_number")
        })

        analysis = await llm.ainvoke(f"""
        다음 소스코드에서 에러 원인을 분석하세요.

        에러 메시지: {state['error_codes']}
        스택 트레이스: {program['stack_trace']}
        소스코드: {source}

        분석 항목:
        1. 에러 발생 원인 (코드 레벨)
        2. 발생 조건 (어떤 상황에서 발생하는가)
        3. 영향 범위
        4. 임시 해결 방법
        5. 근본 수정 방향
        """)
    ...
```

### 6.5 Node 5: answer_generator

```python
async def answer_generator(state: MESAgentState) -> MESAgentState:
    """
    역할: 수집된 모든 정보를 종합하여 최종 답변 생성

    답변 구조:
    1. 요약 (현재 상황 한 줄 정리)
    2. 원인 분석 (MES 데이터 + 로그 + 소스 분석 결과)
    3. 조치 방법 (단계별 가이드)
    4. 추가 참고사항 (관련 문서, 유사 사례)
    5. 후속 조치 필요 여부

    품질 기준:
    - 기술 용어는 사용자 수준에 맞게 조정
    - 근거 데이터 명시 (어떤 로그, 어떤 DB 조회 결과 기반)
    - 불확실한 내용은 명확히 표시
    - 신뢰도 0.4 미만시 에스컬레이션
    """

    context = build_answer_context(state)

    answer_prompt = f"""
    MES LT 시스템 문의에 대한 전문적인 답변을 작성하세요.

    [사용자 질의]
    {state['original_query']}

    [MES 현황 조회 결과]
    {state['mes_query_results']}

    [시스템 로그 분석]
    {format_logs(state['log_entries'])}

    [소스코드 분석] (해당시)
    {state.get('source_analysis', '해당 없음')}

    [관련 문서 및 유사 사례]
    {format_rag_context(state['rag_context'])}

    답변 형식:
    ## 현황 요약
    ## 원인 분석
    ## 조치 방법
    ## 참고 사항
    """
    ...
```

---

## 7. 시스템 컴포넌트 구성

### 7.1 디렉토리 구조

```
mes_ai_agent/
├── main.py                          # 진입점 (FastAPI 서버)
├── config/
│   ├── settings.py                  # 환경 설정
│   └── mcp_config.json              # MCP 서버 설정
├── agent/
│   ├── graph.py                     # LangGraph 워크플로우 정의
│   ├── state.py                     # AgentState 정의
│   ├── nodes/
│   │   ├── query_analyzer.py
│   │   ├── mes_query_executor.py
│   │   ├── log_investigator.py
│   │   ├── source_analyzer.py
│   │   ├── rag_retriever.py
│   │   ├── answer_generator.py
│   │   ├── clarify_user.py
│   │   └── escalate_to_human.py
│   └── edges/
│       └── conditions.py            # 조건부 엣지 로직
├── mcp/
│   ├── mes_db_server.py             # MES DB MCP 서버
│   ├── log_server.py                # 로그 MCP 서버
│   ├── source_server.py             # 소스코드 MCP 서버
│   ├── screen_server.py             # 화면 캡처 MCP 서버
│   └── notify_server.py             # 알림 MCP 서버
├── rag/
│   ├── retriever.py                 # Hybrid RAG 검색기
│   ├── embedder.py                  # 문서 임베딩
│   ├── indexer.py                   # 지식베이스 인덱싱
│   └── knowledge_sources/           # 원천 문서
├── models/
│   └── schemas.py                   # Pydantic 모델
├── api/
│   ├── chat.py                      # 채팅 API 엔드포인트
│   └── webhook.py                   # 웹훅 (슬랙 등)
└── tests/
    ├── test_nodes/
    └── test_integration/
```

### 7.2 기술 스택 상세

| 구분 | 컴포넌트 | 기술 |
|------|----------|------|
| **LLM** | 메인 추론 | Claude claude-sonnet-4-6 (텍스트+이미지) |
| **LLM** | 임베딩 | text-embedding-3-large |
| **워크플로우** | Agent 엔진 | LangGraph 0.2+ |
| **MCP** | 프로토콜 | MCP SDK (Python) |
| **Vector DB** | 지식 저장 | ChromaDB (로컬) / pgvector (운영) |
| **검색** | Sparse | BM25 (rank-bm25) |
| **검색** | Dense | FAISS / ChromaDB |
| **API** | 서버 | FastAPI + WebSocket |
| **DB** | MES 연동 | SQLAlchemy + cx_Oracle |
| **캐시** | 응답 캐시 | Redis |
| **모니터링** | 추적 | LangSmith |

---

## 8. 보안 설계

### 8.1 데이터 접근 제어

```python
class MCSSecurity:
    # DB 접근 제어
    ALLOWED_TABLES = [
        "LOT_MASTER", "LOT_HISTORY", "WIP_STATUS",
        "EQUIP_STATUS", "RECIPE_MASTER", "ALARM_LOG"
        # DML 대상 테이블 제외
    ]
    READ_ONLY = True  # SELECT만 허용
    MAX_ROWS = 1000   # 최대 조회 행수

    # 소스코드 접근 제어
    ALLOWED_MODULES = ["mes-core", "mes-wip", "mes-equip"]
    EXCLUDE_PATTERNS = ["*.properties", "*secret*", "*password*"]

    # 사용자 인증
    def validate_user_permission(self, user_id: str, query_type: str) -> bool:
        # LDAP/AD 연동 권한 확인
        ...
```

### 8.2 개인정보 및 기밀 처리

- 로그에서 개인정보 마스킹 (사번, 이름)
- LLM 전송 데이터에서 민감 정보 제거
- 소스코드 외부 전송 시 익명화
- 답변 생성 후 감사 로그 기록

---

## 9. 성능 설계

### 9.1 응답 시간 목표

| 단계 | 목표 시간 |
|------|----------|
| 질의 분석 | < 3초 |
| MES DB 조회 | < 5초 |
| 로그 조회 | < 5초 |
| 소스 분석 (선택) | < 10초 |
| 답변 생성 | < 5초 |
| **전체 응답** | **< 25초** |

### 9.2 최적화 전략

- **스트리밍 응답**: 답변 생성 시작과 동시에 UI에 스트리밍
- **병렬 처리**: MES DB 조회 + 로그 조회 동시 실행
- **캐싱**: 동일 쿼리 1시간 캐시 (Redis)
- **RAG 사전 필터**: 카테고리 기반 검색 범위 축소

```python
# 병렬 실행 예시 (Node 2 + Node 3 동시 실행)
async def parallel_investigation(state):
    mes_task = asyncio.create_task(mes_query_executor(state))
    log_task = asyncio.create_task(log_investigator(state))
    mes_result, log_result = await asyncio.gather(mes_task, log_task)
    return merge_states(mes_result, log_result)
```

---

## 10. 모니터링 및 품질 관리

### 10.1 메트릭

- **응답 시간**: 각 노드별, 전체 P50/P95/P99
- **답변 품질**: 사용자 피드백 (👍/👎), 에스컬레이션 비율
- **정확도**: 유사 사례 매칭 정확도
- **가용성**: Agent 서버 Uptime

### 10.2 피드백 루프

```
사용자 답변 평가
       │
  ┌────┴────┐
  │ 긍정    │ 부정
  ▼         ▼
성공 사례   실패 사례
DB 저장     분석 →
(RAG 학습)  지식베이스
            업데이트
```

### 10.3 LangSmith 추적

- 모든 LangGraph 실행 추적
- 노드별 입출력 로깅
- LLM 호출 비용 모니터링
- 에러 케이스 수집 및 분석

---

## 11. 구현 로드맵

### Phase 1 (4주): 핵심 기능
- [ ] LangGraph 기본 워크플로우 (5개 노드)
- [ ] MCP mes-db-server 구현
- [ ] MCP log-server 구현
- [ ] 기본 RAG (MES 매뉴얼 인덱싱)
- [ ] FastAPI 채팅 API

### Phase 2 (3주): 고도화
- [ ] 스크린샷 멀티모달 분석
- [ ] MCP source-server 구현
- [ ] 소스코드 분석 노드
- [ ] Hybrid RAG (BM25 + Vector)
- [ ] 과거 장애 이력 인덱싱

### Phase 3 (2주): 운영화
- [ ] 에스컬레이션 워크플로우
- [ ] 슬랙 연동 (MCP notify-server)
- [ ] Redis 캐싱
- [ ] LangSmith 모니터링 연동
- [ ] 부하 테스트 및 최적화

---

## 12. 주요 리스크 및 대응

| 리스크 | 대응 방안 |
|--------|----------|
| MES DB 직접 연결 부하 | Read Replica 사용, 쿼리 캐싱 |
| LLM 환각(Hallucination) | RAG 근거 의무화, 신뢰도 임계값 설정 |
| 로그 데이터 과다 | 로그 필터링 전략, 최대 50건 제한 |
| 소스코드 보안 | 접근 제어 화이트리스트, 감사 로그 |
| 응답 시간 초과 | 타임아웃 설정, 부분 답변 제공 |
| 사용자 개인정보 | 데이터 마스킹, 내부망 LLM 검토 |
