"""
MES AI Agent - FastAPI 애플리케이션 진입점
"""
import logging
import sys
import os

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn

from api.chat import router as chat_router
from config import settings

# ── 로깅 설정 ────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ── FastAPI 앱 생성 ──────────────────────────────────────
app = FastAPI(
    title="MES AI Agent",
    description="반도체 후공정 MES LT 시스템 사용자 문의 응대 AI Agent",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(chat_router, prefix="/api/v1", tags=["Chat"])


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    """간단한 테스트 UI"""
    return """
<!DOCTYPE html>
<html>
<head>
    <title>MES AI Agent</title>
    <meta charset="utf-8">
    <style>
        body { font-family: Arial, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; }
        h1 { color: #2c3e50; }
        #chat-box { border: 1px solid #ddd; height: 400px; overflow-y: auto; padding: 15px;
                    border-radius: 8px; background: #f9f9f9; margin-bottom: 15px; }
        .msg-user { text-align: right; margin: 8px 0; }
        .msg-user span { background: #3498db; color: white; padding: 8px 14px;
                         border-radius: 18px 18px 4px 18px; display: inline-block; max-width: 70%; }
        .msg-agent { text-align: left; margin: 8px 0; }
        .msg-agent span { background: #ecf0f1; padding: 8px 14px;
                          border-radius: 18px 18px 18px 4px; display: inline-block; max-width: 70%;
                          white-space: pre-wrap; }
        .msg-step { color: #7f8c8d; font-size: 12px; margin: 2px 0 2px 10px; }
        #input-area { display: flex; gap: 10px; }
        #query-input { flex: 1; padding: 10px; border: 1px solid #ddd; border-radius: 6px;
                       font-size: 14px; }
        button { padding: 10px 20px; background: #3498db; color: white; border: none;
                 border-radius: 6px; cursor: pointer; font-size: 14px; }
        button:hover { background: #2980b9; }
        .confidence { font-size: 11px; color: #95a5a6; margin-top: 4px; }
    </style>
</head>
<body>
    <h1>🏭 MES AI Agent</h1>
    <p>반도체 후공정 MES LT 시스템 문의 응대 AI</p>
    <div id="chat-box"></div>
    <div id="input-area">
        <input id="query-input" type="text" placeholder="MES 관련 문의를 입력하세요... (예: LOT-2024-001이 왜 HOLD 됐나요?)"
               onkeydown="if(event.key==='Enter') sendMessage()">
        <button onclick="sendMessage()">전송</button>
    </div>

    <script>
        let ws = null;
        const chatBox = document.getElementById('chat-box');
        const sessionId = 'session-' + Date.now();

        function connect() {
            ws = new WebSocket('ws://' + location.host + '/api/v1/ws/chat');
            ws.onmessage = (e) => {
                const msg = JSON.parse(e.data);
                handleMessage(msg);
            };
            ws.onclose = () => { setTimeout(connect, 2000); };
        }

        function handleMessage(msg) {
            if (msg.type === 'progress' || msg.type === 'status') {
                const div = document.createElement('div');
                div.className = 'msg-step';
                div.textContent = msg.step || msg.message;
                chatBox.appendChild(div);
            } else if (msg.type === 'answer') {
                const div = document.createElement('div');
                div.className = 'msg-agent';
                div.innerHTML = '<span>' + msg.answer.replace(/</g,'&lt;').replace(/>/g,'&gt;') + '</span>'
                    + '<div class="confidence">신뢰도: ' + (msg.confidence * 100).toFixed(1) + '%'
                    + (msg.escalation_required ? ' | ⚠️ 에스컬레이션: ' + msg.escalation_level : '') + '</div>';
                chatBox.appendChild(div);
            } else if (msg.type === 'clarification') {
                const div = document.createElement('div');
                div.className = 'msg-agent';
                div.innerHTML = '<span>❓ ' + msg.message + '</span>';
                chatBox.appendChild(div);
            } else if (msg.type === 'error') {
                const div = document.createElement('div');
                div.style.color = 'red';
                div.textContent = '오류: ' + msg.message;
                chatBox.appendChild(div);
            }
            chatBox.scrollTop = chatBox.scrollHeight;
        }

        function sendMessage() {
            const input = document.getElementById('query-input');
            const message = input.value.trim();
            if (!message || !ws || ws.readyState !== WebSocket.OPEN) return;

            const div = document.createElement('div');
            div.className = 'msg-user';
            div.innerHTML = '<span>' + message + '</span>';
            chatBox.appendChild(div);

            ws.send(JSON.stringify({
                session_id: sessionId,
                user_id: 'test-user',
                message: message,
            }));
            input.value = '';
        }

        connect();
    </script>
</body>
</html>
"""


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "mes-ai-agent"}


if __name__ == "__main__":
    logger.info(f"MES AI Agent 시작: {settings.api_host}:{settings.api_port}")
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level="info",
    )
