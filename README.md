# MES AI Agent

반도체 후공정 MES(Manufacturing Execution System) 전문가용 AI 에이전트입니다. 사용자의 질의(텍스트 및 스크린샷)를 분석하여 데이터 오류, 설비 이상, 로그 분석 및 소스 코드 추적을 통해 정확한 답변과 대응 방안을 제공합니다.

## 🚀 주요 기능

- **지능형 질의 분석 (Multimodal)**: OpenAI Vision을 활용하여 텍스트 질의와 첨부된 스크린샷을 동시에 분석합니다.
- **Human-in-the-Loop (HITL)**: 부족한 정보가 있을 경우 LangGraph의 `interrupt` 기능을 통해 사용자에게 추가 질문을 던지고, 답변을 받아 처리를 재개합니다.
- **LLM 기반 도구 선택 (Function Calling)**: MCP(Model Context Protocol) 기반의 다양한 도구(MES DB, 로그 검색 등)를 LLM이 의도에 따라 동적으로 선택하여 실행합니다.
- **복합 원인 분석**:
    - **MES DB**: Lot, 설비, 공정 등의 실시간 상태 조회
    - **Elasticsearch Log**: 시스템 에러 로그 및 프로그램 이슈 추적
    - **Source Code**: 실제 구현 코드 로직 분석 지원
    - **RAG (Retrieval-Augmented Generation)**: MES 매뉴얼 및 유사 장애 사례 기반 탐색
- **중앙 집중식 SQL 관리**: 모든 비즈니스 쿼리를 `sql_queries.py`에서 통합 관리하여 유지보수성을 높였습니다.

## 🛠 기술 스택

- **Framework**: LangGraph, LangChain
- **LLM**: OpenAI GPT-4o
- **API**: FastAPI, WebSocket
- **Integrations**: MCP (Model Context Protocol), SQLAlchemy, Elasticsearch, ChromaDB (RAG)
- **Language**: Python 3.10+

## 📁 프로젝트 구조

```text
mes_ai_agent/
├── agent/              # LangGraph 워크플로우 및 노드 정의
│   ├── nodes/          # Query Analyzer, Executor, Generator 등 각 단계별 로직
│   └── graph.py        # 전체 Agent 상태 그래프 구성
├── api/                # FastAPI 엔드포인트 (Chat, WebSocket)
├── config/             # 환경 설정 및 Settings 관리
├── servers/            # MCP 도구 서버 및 유틸리티
├── rag/                # RAG 데이터 인덱싱 및 검색 로직
├── models/             # 데이터 스키마 (Pydantic, SQLAlchemy)
├── docs/               # 설계 문서 및 기술 가이드
├── static/             # 목데이터 및 정적 자산 (JSON 등)
└── main.py             # 애플리케이션 실행 진입점
```

## ⚙️ 설치 및 실행 방법

### 1. 환경 설정
`.env` 파일을 루트 디렉토리에 생성하고 필요한 API 키와 DB 정보를 입력합니다. (참고: `.env.example`)

```bash
OPENAI_API_KEY=your_api_key_here
MES_DB_URL=postgresql+psycopg2://user:password@host:port/dbname
ES_HOST=http://your-es-host:9200
```

### 2. 패키지 설치
```bash
pip install -r requirements.txt
```

### 3. 서버 실행
```bash
python main.py
```
서버가 시작되면 `http://localhost:8000/docs`에서 API 명세(Swagger UI)를 확인할 수 있습니다.

## 🧪 테스트 및 검증
통합 테스트 및 HITL 흐름을 검증할 수 있는 스크립트가 포함되어 있습니다.

```bash
# 전체 노드 및 그래프 테스트
pytest tests/test_nodes/test_graph.py

# Human-in-the-Loop 흐름 검증
python tests/verify_hitl.py
```

## 📝 라이선스
이 프로젝트는 개인/교육용으로 작성되었습니다.
