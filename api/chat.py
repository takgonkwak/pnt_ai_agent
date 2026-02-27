"""
FastAPI 채팅 API
- POST /chat: 일반 질의 응답 (interrupt 지원: 재문의 시 needs_clarification=True 반환)
- WebSocket /ws/chat: 실시간 스트리밍 응답 (interrupt 자동 감지 및 재개)
- GET /tickets: 에스컬레이션 티켓 목록
"""
import uuid
import json
import logging
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse
from langgraph.types import Command

from agent.graph import get_graph
from agent.state import create_initial_state
from models.schemas import ChatRequest, ChatResponse, EscalationLevel
from servers.notify_server import list_open_tickets, get_ticket

logger = logging.getLogger(__name__)

router = APIRouter()


def _thread_config(session_id: str) -> dict:
    """LangGraph 체크포인터 thread 설정"""
    return {"configurable": {"thread_id": session_id}}


@router.post("/chat", response_model=ChatResponse, summary="MES 문의 응답")
async def chat(request: ChatRequest):
    """
    MES 시스템 사용자 문의 처리 (동기 응답)

    - 재문의 필요 시: needs_clarification=True, clarification_question 반환
    - 클라이언트는 같은 session_id로 재호출하면 그래프가 중단 지점부터 재개됨
    """
    logger.info(f"[API] chat: user={request.user_id}, session={request.session_id}")

    try:
        graph = get_graph()
        config = _thread_config(request.session_id)

        # 기존 스레드 상태 확인 (interrupt 대기 중인지 확인)
        snapshot = await graph.aget_state(config)
        if snapshot.tasks:
            # 재문의 답변: 일시 정지된 그래프 재개
            logger.info(f"[API] 재문의 재개: session={request.session_id}")
            input_data = Command(resume=request.message)
        else:
            # 새 질의: 초기 상태로 시작
            input_data = create_initial_state(
                user_id=request.user_id,
                session_id=request.session_id,
                query=request.message,
                screenshots=request.screenshots,
            )

        final_state = await graph.ainvoke(input_data, config=config)

        # interrupt 발생 여부 확인 (clarify_user 노드에서 pause)
        new_snapshot = await graph.aget_state(config)
        if new_snapshot.tasks:
            interrupt_val = new_snapshot.tasks[0].interrupts[0].value
            clarification_q = interrupt_val.get("question", "")
            logger.info(f"[API] 재문의 필요: {clarification_q[:60]}")
            return ChatResponse(
                session_id=request.session_id,
                answer=clarification_q,
                confidence=0.0,
                escalation_required=False,
                escalation_level=EscalationLevel.NONE,
                processing_steps=final_state.get("processing_steps", []) if isinstance(final_state, dict) else [],
                needs_clarification=True,
                clarification_question=clarification_q,
            )

        return ChatResponse(
            session_id=request.session_id,
            answer=final_state.get("final_answer", "답변을 생성할 수 없습니다."),
            confidence=final_state.get("answer_confidence", 0.0),
            escalation_required=final_state.get("escalation_required", False),
            escalation_level=final_state.get("escalation_level", EscalationLevel.NONE),
            processing_steps=final_state.get("processing_steps", []),
            sources=[
                {"source": doc.get("source", ""), "content": doc.get("content", "")[:100]}
                for doc in final_state.get("rag_context", [])[:3]
            ],
        )

    except Exception as e:
        logger.error(f"[API] chat 오류: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agent 처리 오류: {str(e)}")


@router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """
    WebSocket 실시간 채팅

    처리 단계를 실시간으로 클라이언트에 스트리밍.
    interrupt 발생 시 재문의 질문 전송 후 대기, 다음 메시지로 자동 재개.
    """
    await websocket.accept()
    logger.info("[API] WebSocket 연결")

    try:
        graph = get_graph()

        while True:
            # 클라이언트 메시지 수신
            data = await websocket.receive_text()
            payload = json.loads(data)

            session_id = payload.get("session_id", str(uuid.uuid4()))
            user_id = payload.get("user_id", "anonymous")
            message = payload.get("message", "")
            screenshots = payload.get("screenshots", [])
            config = _thread_config(session_id)

            if not message:
                await websocket.send_json({"type": "error", "message": "메시지가 비어있습니다."})
                continue

            # interrupt 대기 중인 스레드인지 확인
            snapshot = await graph.aget_state(config)
            if snapshot.tasks:
                # 재문의 답변 → Command(resume=...) 로 그래프 재개
                logger.info(f"[API] WS 재문의 재개: session={session_id}")
                input_data = Command(resume=message)
            else:
                # 새 질의 → 초기 상태 생성
                input_data = create_initial_state(
                    user_id=user_id,
                    session_id=session_id,
                    query=message,
                    screenshots=screenshots,
                )
                await websocket.send_json({
                    "type": "status",
                    "message": "🔍 질의를 분석하고 있습니다..."
                })

            # LangGraph 스트리밍 실행
            interrupted = False
            async for chunk in graph.astream(input_data, config=config, stream_mode="updates"):
                if "__interrupt__" in chunk:
                    # clarify_user 노드에서 interrupt() 호출됨
                    interrupt_val = chunk["__interrupt__"][0].value
                    clarification_q = interrupt_val.get("question", "")
                    logger.info(f"[API] WS interrupt: {clarification_q[:60]}")
                    await websocket.send_json({
                        "type": "clarification",
                        "message": clarification_q,
                    })
                    interrupted = True
                else:
                    for node_name, node_state in chunk.items():
                        steps = node_state.get("processing_steps", [])
                        if steps:
                            await websocket.send_json({
                                "type": "progress",
                                "node": node_name,
                                "step": steps[-1],
                            })

            if not interrupted:
                # 정상 완료: 최종 상태 조회 후 답변 전송
                final_snapshot = await graph.aget_state(config)
                fv = final_snapshot.values
                await websocket.send_json({
                    "type": "answer",
                    "session_id": session_id,
                    "answer": fv.get("final_answer", ""),
                    "confidence": fv.get("answer_confidence", 0.0),
                    "escalation_required": fv.get("escalation_required", False),
                    "escalation_level": fv.get("escalation_level", "NONE"),
                    "processing_steps": fv.get("processing_steps", []),
                })

    except WebSocketDisconnect:
        logger.info("[API] WebSocket 연결 해제")
    except Exception as e:
        logger.error(f"[API] WebSocket 오류: {e}", exc_info=True)
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


@router.get("/tickets", summary="에스컬레이션 티켓 목록")
async def get_tickets():
    """오픈된 에스컬레이션 티켓 목록 조회"""
    tickets = list_open_tickets()
    return JSONResponse({"tickets": tickets, "count": len(tickets)})


@router.get("/tickets/{ticket_id}", summary="티켓 상세 조회")
async def get_ticket_detail(ticket_id: str):
    """특정 티켓 상세 조회"""
    ticket = get_ticket(ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail=f"티켓 {ticket_id}을 찾을 수 없습니다.")
    return JSONResponse(ticket)
