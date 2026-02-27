"""
RAG 지식베이스 인덱서
MES 매뉴얼, 장애 이력, 기술 문서를 청크로 분할하고 벡터 DB에 저장
"""
import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# 문서 청킹 설정
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64


class KnowledgeIndexer:
    """지식베이스 문서 인덱서"""

    def __init__(self, retriever=None):
        self.retriever = retriever

    async def index_directory(
        self,
        directory: str,
        category: str = "manual",
        file_extensions: List[str] = None,
    ):
        """디렉토리 내 문서 전체 인덱싱"""
        if file_extensions is None:
            file_extensions = [".txt", ".md", ".json"]

        path = Path(directory)
        if not path.exists():
            logger.warning(f"[Indexer] 경로 없음: {directory}")
            return

        indexed = 0
        for file_path in path.rglob("*"):
            if file_path.suffix.lower() in file_extensions:
                docs = self._load_and_chunk(str(file_path), category)
                if docs and self.retriever:
                    await self.retriever.add_documents(docs)
                indexed += len(docs)

        logger.info(f"[Indexer] 인덱싱 완료: {directory}, {indexed}개 청크")

    def _load_and_chunk(
        self,
        file_path: str,
        category: str,
    ) -> List[Dict[str, Any]]:
        """파일 로드 및 청킹"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            if file_path.endswith(".json"):
                # JSON 형식 (장애 이력 등)
                data = json.loads(content)
                if isinstance(data, list):
                    return [
                        {
                            "doc_id": f"{Path(file_path).stem}-{i}",
                            "source": Path(file_path).name,
                            "content": json.dumps(item, ensure_ascii=False),
                            "category": category,
                            "metadata": {"file": file_path},
                        }
                        for i, item in enumerate(data)
                    ]

            # 텍스트/마크다운: 청크 분할
            chunks = self._split_text(content, CHUNK_SIZE, CHUNK_OVERLAP)
            docs = []
            for i, chunk in enumerate(chunks):
                docs.append({
                    "doc_id": f"{Path(file_path).stem}-chunk{i}",
                    "source": Path(file_path).name,
                    "content": chunk,
                    "category": category,
                    "metadata": {"file": file_path, "chunk_index": i},
                })
            return docs

        except Exception as e:
            logger.error(f"[Indexer] 파일 로드 오류 {file_path}: {e}")
            return []

    def _split_text(
        self,
        text: str,
        chunk_size: int,
        overlap: int,
    ) -> List[str]:
        """텍스트 청킹 (문단 경계 존중)"""
        paragraphs = text.split("\n\n")
        chunks = []
        current = ""

        for para in paragraphs:
            if len(current) + len(para) < chunk_size:
                current += para + "\n\n"
            else:
                if current:
                    chunks.append(current.strip())
                    # 오버랩: 마지막 overlap 글자 유지
                    current = current[-overlap:] + para + "\n\n"
                else:
                    # 단일 문단이 chunk_size 초과 시 강제 분할
                    for j in range(0, len(para), chunk_size - overlap):
                        chunks.append(para[j:j + chunk_size])
                    current = ""

        if current.strip():
            chunks.append(current.strip())

        return [c for c in chunks if len(c) > 20]
