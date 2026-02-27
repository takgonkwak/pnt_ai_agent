"""
FastMCP Server: source-server
MES 소스코드 조회 (클래스/메서드/라인 기반)

실제 환경: GitHub REST API (SOURCE_ENABLED=true)
개발/테스트: 샘플 Java 소스 (SOURCE_ENABLED=false)

GitHub API 사용 흐름:
  get_source_by_class:
    1. Code Search API 로 클래스 파일 경로 탐색
       GET /search/code?q=class+{ClassName}+in:file+language:Java+repo:{owner}/{repo}
    2. Contents API 로 파일 원문 조회
       GET /repos/{owner}/{repo}/contents/{path}?ref={branch}
    3. line_number 지정 시 ±context_lines 범위 추출

  search_code:
    GitHub Code Search API
    GET /search/code?q={query}+repo:{owner}/{repo}
"""
import base64
import logging
from typing import Optional

import httpx
from fastmcp import FastMCP
from config import settings

logger = logging.getLogger(__name__)

source_mcp = FastMCP("source-server")

_GH_API = "https://api.github.com"

# ── Mock 샘플 데이터 ──────────────────────────────────────

SAMPLE_SOURCES = {
    "WireBondService": """package com.mes.equipment;

import com.mes.exception.ValidationException;
import com.mes.model.Recipe;
import com.mes.model.Lot;

@Service
public class WireBondService {

    @Autowired
    private RecipeRepository recipeRepo;

    @Autowired
    private LotRepository lotRepo;

    public void processLot(String lotId, String equipId) {
        Lot lot = lotRepo.findById(lotId)
            .orElseThrow(() -> new IllegalArgumentException("Lot not found: " + lotId));

        Recipe recipe = recipeRepo.findByEquipAndProduct(equipId, lot.getProductId())
            .orElseThrow(() -> new IllegalArgumentException("Recipe not found"));

        // 와이어 장력 검증 (line 245 에서 오류 발생)
        validateTension(equipId, recipe);   // line 245

        lot.setStatus("RUN");
        lotRepo.save(lot);
    }

    public void validateTension(String equipId, Recipe recipe) {  // line 280
        double currentTension = getEquipTension(equipId);
        double maxTension = recipe.getMaxWireTension();

        // BUG: recipe.getMaxWireTension()이 null 반환 시 NullPointerException 발생
        // 단위 불일치: 설비는 g, 레시피는 cN (1g = 0.981cN)
        if (currentTension > maxTension) {  // line 312 - 오류 발생 지점
            throw new ValidationException(
                String.format("Wire tension out of range (설정: %.1fg, 현재: %.1fg)",
                    maxTension, currentTension)
            );
        }
    }

    private double getEquipTension(String equipId) {
        return equipSensorService.readTension(equipId);
    }
}
""",
}


# ── MCP Tools ─────────────────────────────────────────────

@source_mcp.tool()
async def get_source_by_class(
    class_name: str,
    method_name: Optional[str] = None,
    line_number: Optional[int] = None,
    context_lines: int = 30,
) -> Optional[str]:
    """
    로그에 표시된 클래스명으로 GitHub 소스코드 조회

    Args:
        class_name:    Java 클래스명 (단순명 또는 FQN)
        method_name:   조회할 메서드명 (미지정 시 파일 전체 반환)
        line_number:   오류 발생 라인 번호 (지정 시 ±context_lines 범위 추출)
        context_lines: 라인 기준 위아래 포함 줄 수 (기본 30)
    """
    logger.info(f"[source] get_source_by_class: {class_name}.{method_name}:{line_number}")

    if not settings.source_enabled:
        return _mock_source(class_name)

    return await _github_get_source(class_name, method_name, line_number, context_lines)


@source_mcp.tool()
async def search_code(
    query: str,
    file_pattern: str = "*.java",
    module: Optional[str] = None,
) -> list:
    """
    GitHub 저장소에서 특정 패턴 또는 에러 코드 검색

    Args:
        query:        검색 쿼리 (에러 코드, 메서드명, 예외 클래스 등)
        file_pattern: 대상 파일 패턴 (기본 *.java)
        module:       특정 모듈/디렉토리로 범위 제한 (예: "equipment")
    """
    logger.info(f"[source] search_code: '{query}', pattern={file_pattern}, module={module}")

    if not settings.source_enabled:
        return _mock_search(query)

    return await _github_search_code(query, file_pattern, module)


# ── Mock 헬퍼 ─────────────────────────────────────────────

def _mock_source(class_name: str) -> Optional[str]:
    src = SAMPLE_SOURCES.get(class_name)
    if not src:
        src = next((v for k, v in SAMPLE_SOURCES.items()
                    if class_name.lower() in k.lower()), None)
    return src


def _mock_search(query: str) -> list:
    results = []
    for cls, src in SAMPLE_SOURCES.items():
        for i, line in enumerate(src.split("\n"), 1):
            if query.lower() in line.lower():
                results.append({"class_name": cls, "line_number": i,
                                "line_content": line.strip()})
    return results


# ── GitHub API 헬퍼 ──────────────────────────────────────

def _gh_headers() -> dict:
    """GitHub REST API 공통 헤더"""
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if settings.github_token:
        headers["Authorization"] = f"Bearer {settings.github_token}"
    return headers


async def _github_get_source(
    class_name: str,
    method_name: Optional[str],
    line_number: Optional[int],
    context_lines: int,
) -> Optional[str]:
    """
    1단계: Code Search로 파일 경로 확인
    2단계: Contents API로 파일 원문 조회
    3단계: line_number 지정 시 범위 추출
    """
    # 단순 클래스명 추출 (FQN 대응: com.mes.equipment.WireBondService → WireBondService)
    simple_name = class_name.split(".")[-1]

    async with httpx.AsyncClient(headers=_gh_headers(), timeout=15.0) as client:
        # 1단계: Code Search
        file_path = await _search_class_path(client, simple_name)
        if not file_path:
            logger.warning(f"[source] GitHub에서 {simple_name}.java 파일을 찾지 못했습니다")
            return None

        # 2단계: 파일 원문 조회
        content = await _fetch_file_content(client, file_path)
        if not content:
            return None

    # 3단계: 라인 범위 추출
    if line_number:
        lines = content.split("\n")
        start = max(0, line_number - context_lines - 1)
        end = min(len(lines), line_number + context_lines)
        return "\n".join(lines[start:end])

    return content


async def _search_class_path(client: httpx.AsyncClient, simple_name: str) -> Optional[str]:
    """GitHub Code Search로 Java 클래스 파일 경로 반환"""
    q = f"class {simple_name} in:file language:Java repo:{settings.github_owner}/{settings.github_repo}"
    resp = await client.get(f"{_GH_API}/search/code", params={"q": q, "per_page": 5})

    if resp.status_code != 200:
        logger.error(f"[source] Code Search 실패: {resp.status_code} {resp.text[:200]}")
        return None

    items = resp.json().get("items", [])
    if not items:
        return None

    # 파일명이 정확히 일치하는 항목 우선 선택
    exact = next((i for i in items if i["name"] == f"{simple_name}.java"), None)
    return (exact or items[0])["path"]


async def _fetch_file_content(client: httpx.AsyncClient, path: str) -> Optional[str]:
    """GitHub Contents API로 파일 원문(UTF-8) 반환"""
    url = f"{_GH_API}/repos/{settings.github_owner}/{settings.github_repo}/contents/{path}"
    resp = await client.get(url, params={"ref": settings.github_branch})

    if resp.status_code != 200:
        logger.error(f"[source] Contents API 실패: {resp.status_code} path={path}")
        return None

    data = resp.json()
    encoded = data.get("content", "")
    try:
        return base64.b64decode(encoded).decode("utf-8")
    except Exception as e:
        logger.error(f"[source] base64 디코딩 오류: {e}")
        return None


async def _github_search_code(query: str, file_pattern: str, module: Optional[str]) -> list:
    """GitHub Code Search API로 코드 내 키워드 검색"""
    ext = file_pattern.replace("*", "").lstrip(".")   # "*.java" → "java"
    q = f"{query} language:{ext} repo:{settings.github_owner}/{settings.github_repo}"
    if module:
        q += f" path:{module}"

    async with httpx.AsyncClient(
        headers={**_gh_headers(),
                 "Accept": "application/vnd.github.v3.text-match+json"},
        timeout=15.0,
    ) as client:
        resp = await client.get(f"{_GH_API}/search/code", params={"q": q, "per_page": 20})

    if resp.status_code != 200:
        logger.error(f"[source] Code Search 실패: {resp.status_code}")
        return []

    results = []
    for item in resp.json().get("items", []):
        entry = {
            "path": item.get("path", ""),
            "class_name": item.get("name", "").replace(".java", ""),
            "html_url": item.get("html_url", ""),
            "matches": [],
        }
        for tm in item.get("text_matches", []):
            for frag in tm.get("matches", []):
                entry["matches"].append(frag.get("text", ""))
        results.append(entry)

    return results
