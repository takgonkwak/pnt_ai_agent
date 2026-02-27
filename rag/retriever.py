"""
RAG 검색 엔진
Hybrid 검색: Dense (ChromaDB 벡터) + Sparse (BM25 키워드)
RRF(Reciprocal Rank Fusion)로 결과 통합
"""
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RAGResult:
    """RAG 검색 결과 단위"""
    doc_id: str
    source: str
    content: str
    category: str       # manual | incident | technical
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


# 샘플 지식 베이스 (초기화 전 테스트용)
SAMPLE_KNOWLEDGE = [
    {
        "doc_id": "KB-001",
        "source": "MES_매뉴얼_와이어본딩.pdf",
        "content": (
            "와이어본딩 공정에서 Wire Tension(와이어 장력) 알람이 발생하면 "
            "다음 절차를 따르세요:\n"
            "1. 해당 설비를 즉시 정지하고 HOLD 처리\n"
            "2. 레시피의 Wire Tension 설정값과 현재 센서 값 비교\n"
            "3. 캐필러리(Capillary) 교체 여부 확인 (PM 주기: 5만 타)\n"
            "4. 와이어 스풀 장력 조정 나사 확인\n"
            "5. 설비 엔지니어에게 보고"
        ),
        "category": "manual",
        "metadata": {"equip_type": "WIRE_BONDER", "error_code": "ALM-WB-001"}
    },
    {
        "doc_id": "KB-002",
        "source": "장애이력_2023_Q4.xlsx",
        "content": (
            "[장애 사례 #2023-1105]\n"
            "증상: EQUIP-WB01에서 LOT HOLD 반복 발생\n"
            "원인: 레시피 RECIPE-WB-A100-V1의 MaxWireTension 값이 null로 저장됨 (DB 마이그레이션 오류)\n"
            "조치: DB UPDATE로 MaxWireTension = 12 설정, WireBondService.validateTension() 버그 수정\n"
            "수정: null 체크 로직 추가, 단위 변환(g↔cN) 처리 추가\n"
            "재발 방지: 레시피 등록 시 필수값 검증 강화"
        ),
        "category": "incident",
        "metadata": {"equip_id": "EQUIP-WB01", "error_code": "MES-ERR-001"}
    },
    {
        "doc_id": "KB-003",
        "source": "MES_에러코드_정의서.xlsx",
        "content": (
            "에러코드: MES-ERR-001\n"
            "설명: Lot 처리 중 검증 실패\n"
            "발생 조건: 설비 파라미터가 레시피 허용 범위 초과\n"
            "조치 방법: (1) 설비 상태 확인, (2) 레시피 파라미터 검토, (3) 설비 엔지니어 연락\n"
            "에스컬레이션: L2 (시스템 개발자 확인 필요 시)"
        ),
        "category": "technical",
        "metadata": {"error_code": "MES-ERR-001"}
    },
    {
        "doc_id": "KB-004",
        "source": "MES_매뉴얼_Lot관리.pdf",
        "content": (
            "Lot HOLD 해제 절차:\n"
            "1. MES > Lot 관리 > Lot 조회 화면에서 해당 Lot 선택\n"
            "2. [HOLD 해제] 버튼 클릭 (권한: 공정 담당자 이상)\n"
            "3. HOLD 사유 확인 및 조치 완료 확인서 첨부\n"
            "4. 담당 엔지니어 승인 후 공정 재개\n"
            "주의: HOLD 해제 전 반드시 원인 해결 확인 필요"
        ),
        "category": "manual",
        "metadata": {"process": "LOT_HOLD", "screen": "LOT_MGMT"}
    },
    {
        "doc_id": "KB-005",
        "source": "시스템아키텍처_v2.3.pdf",
        "content": (
            "WireBondService 클래스 설계:\n"
            "- 패키지: com.mes.equipment\n"
            "- 의존성: RecipeRepository, LotRepository, EquipSensorService\n"
            "- 주요 메서드: processLot(), validateTension(), getEquipTension()\n"
            "- validateTension(): 레시피의 MaxWireTension과 현재 설비 센서값 비교\n"
            "  주의: recipe.getMaxWireTension() null 시 NPE 발생 가능 (Known Bug)"
        ),
        "category": "technical",
        "metadata": {"class": "WireBondService", "module": "equipment"}
    },
]


class MESRAGRetriever:
    """
    MES 지식베이스 Hybrid 검색기
    Dense (ChromaDB) + Sparse (BM25) → RRF 통합
    """

    def __init__(self):
        self._vector_store = None
        self._bm25 = None
        self._documents = SAMPLE_KNOWLEDGE.copy()
        self._initialized = False

    def _ensure_initialized(self):
        """지연 초기화 (첫 검색 시 실행)"""
        if self._initialized:
            return
        self._init_bm25()
        self._initialized = True

    def _init_bm25(self):
        """BM25 인덱스 초기화"""
        try:
            from rank_bm25 import BM25Okapi
            tokenized = [
                self._tokenize(doc["content"] + " " + doc.get("source", ""))
                for doc in self._documents
            ]
            self._bm25 = BM25Okapi(tokenized)
            logger.info(f"[RAG] BM25 초기화: {len(self._documents)}개 문서")
        except ImportError:
            logger.warning("[RAG] rank_bm25 미설치, Sparse 검색 비활성화")
            self._bm25 = None

    def _tokenize(self, text: str) -> List[str]:
        """간단한 토크나이즈 (공백/특수문자 분리)"""
        import re
        return re.sub(r'[^\w\s]', ' ', text.lower()).split()

    async def _init_vector_store(self):
        """ChromaDB 벡터 스토어 초기화"""
        try:
            import chromadb
            from langchain_chroma import Chroma
            from langchain_openai import OpenAIEmbeddings
            from config import settings

            embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                api_key=settings.openai_api_key,
            )
            self._vector_store = Chroma(
                collection_name=settings.chroma_collection_name,
                embedding_function=embeddings,
                persist_directory=settings.chroma_persist_dir,
            )
            logger.info("[RAG] ChromaDB 초기화 완료")
        except Exception as e:
            logger.warning(f"[RAG] ChromaDB 초기화 실패 (Fallback to BM25only): {e}")
            self._vector_store = None

    async def retrieve(
        self,
        query: str,
        context: Dict[str, Any],
        k: int = 5,
    ) -> List[RAGResult]:
        """
        Hybrid 검색 실행
        1. BM25 Sparse 검색
        2. Dense 검색 (ChromaDB, 설정된 경우)
        3. RRF 통합
        4. 카테고리 필터 적용
        """
        self._ensure_initialized()

        # 카테고리 필터
        category_filter = context.get("category_filter")
        docs = self._documents
        if category_filter:
            docs = [d for d in docs if d.get("category") == category_filter]

        if not docs:
            return []

        # BM25 검색
        bm25_results = self._bm25_search(query, docs, k * 2)

        # Dense 검색 (ChromaDB)
        dense_results = await self._dense_search(query, k * 2)

        # RRF 통합
        if dense_results:
            merged = self._reciprocal_rank_fusion(bm25_results, dense_results, k)
        else:
            merged = bm25_results[:k]

        logger.info(f"[RAG] 검색 완료: query='{query[:30]}', 결과={len(merged)}건")
        return merged

    def _bm25_search(
        self,
        query: str,
        docs: List[Dict],
        k: int,
    ) -> List[RAGResult]:
        """BM25 키워드 검색"""
        if not self._bm25:
            # BM25 없으면 키워드 매칭으로 대체
            return self._keyword_search(query, docs, k)

        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        # 점수 + 인덱스 정렬
        scored = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )[:k]

        results = []
        for idx, score in scored:
            if idx < len(self._documents) and score > 0:
                doc = self._documents[idx]
                results.append(RAGResult(
                    doc_id=doc["doc_id"],
                    source=doc["source"],
                    content=doc["content"],
                    category=doc["category"],
                    score=float(score),
                    metadata=doc.get("metadata", {}),
                ))
        return results

    def _keyword_search(
        self,
        query: str,
        docs: List[Dict],
        k: int,
    ) -> List[RAGResult]:
        """단순 키워드 매칭 (fallback)"""
        query_lower = query.lower()
        keywords = query_lower.split()
        results = []

        for doc in docs:
            text = (doc["content"] + " " + doc.get("source", "")).lower()
            score = sum(1 for kw in keywords if kw in text) / max(len(keywords), 1)
            if score > 0:
                results.append(RAGResult(
                    doc_id=doc["doc_id"],
                    source=doc["source"],
                    content=doc["content"],
                    category=doc["category"],
                    score=score,
                    metadata=doc.get("metadata", {}),
                ))

        return sorted(results, key=lambda x: x.score, reverse=True)[:k]

    async def _dense_search(self, query: str, k: int) -> List[RAGResult]:
        """Dense 벡터 검색 (ChromaDB)"""
        if self._vector_store is None:
            return []
        try:
            docs_with_scores = self._vector_store.similarity_search_with_score(query, k=k)
            results = []
            for doc, score in docs_with_scores:
                results.append(RAGResult(
                    doc_id=doc.metadata.get("doc_id", ""),
                    source=doc.metadata.get("source", ""),
                    content=doc.page_content,
                    category=doc.metadata.get("category", "manual"),
                    score=float(1 - score),  # 거리 → 유사도
                    metadata=doc.metadata,
                ))
            return results
        except Exception as e:
            logger.error(f"[RAG] Dense 검색 오류: {e}")
            return []

    def _reciprocal_rank_fusion(
        self,
        bm25_results: List[RAGResult],
        dense_results: List[RAGResult],
        k: int,
        rrf_k: int = 60,
    ) -> List[RAGResult]:
        """RRF(Reciprocal Rank Fusion)로 두 결과 통합"""
        scores: Dict[str, float] = {}
        doc_map: Dict[str, RAGResult] = {}

        for rank, result in enumerate(bm25_results):
            scores[result.doc_id] = scores.get(result.doc_id, 0) + 1 / (rrf_k + rank + 1)
            doc_map[result.doc_id] = result

        for rank, result in enumerate(dense_results):
            scores[result.doc_id] = scores.get(result.doc_id, 0) + 1 / (rrf_k + rank + 1)
            doc_map[result.doc_id] = result

        sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)[:k]
        results = []
        for doc_id in sorted_ids:
            result = doc_map[doc_id]
            result.score = scores[doc_id]
            results.append(result)

        return results

    async def add_documents(self, documents: List[Dict[str, Any]]):
        """지식베이스에 문서 추가"""
        self._documents.extend(documents)
        self._init_bm25()  # BM25 재인덱싱

        if self._vector_store:
            from langchain_core.documents import Document
            lc_docs = [
                Document(
                    page_content=doc["content"],
                    metadata={
                        "doc_id": doc["doc_id"],
                        "source": doc["source"],
                        "category": doc["category"],
                        **doc.get("metadata", {}),
                    }
                )
                for doc in documents
            ]
            self._vector_store.add_documents(lc_docs)

        logger.info(f"[RAG] {len(documents)}개 문서 추가 (총 {len(self._documents)}개)")
