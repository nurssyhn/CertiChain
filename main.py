from fastapi import FastAPI, WebSocket,WebSocketDisconnect
from fastapi.responses import HTMLResponse
from llm import rag_chain, HumanMessage, SystemMessage

app = FastAPI()

html = """

<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IT Serv. AI BOT</title>
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Roboto', sans-serif;
            max-width: 600px;
            margin: 0 auto;
            padding: 20px;
        }
        h1 {
            font-weight: 700;
            font-size:22px;
        }
        #chat-container {
            height: 400px;
            border-radius:8px;
            border: 1px solid #ccc;
            overflow-y: auto;
            padding: 10px;
            margin-bottom: 20px;
        }
        .message {
            margin: 10px 0;
            padding: 8px 12px;
            border-radius: 20px;
            max-width: 70%;
            font-size:14px;
            word-wrap: break-word;
        }
        .user-message {
            background-color: #007bff;
            color: white;
            margin-left: auto;
        }
        .bot-message {
            background-color: #f1f0f0;
            color: black;
        }
        #message-form {
            display: flex;
        }
        #message-input {
            flex: 1;
            outline: none;
            border: solid 1px #33;
            padding: 10px;
            font-size: 13px;
            font-family: 'Roboto', sans-serif;
        }
        button {
            padding: 10px 20px;
            font-size: 14px;
            background-color: #007bff;
            color: white;
            border: none;
            cursor: pointer;
            font-family: 'Roboto', sans-serif;
            font-weight: 700;
        }
        #disclaimer {
            font-size: 12px;
            color: #666;
            text-align: center;
            margin-top: 10px;
        }
    </style>
</head>
<body>
    <h1 style="text-align: center;">ServAI</h1>
    <div id="chat-container"></div>
    <form id="message-form">
        <input type="text" id="message-input" autocomplete="off" placeholder="ServAI ile mesajlaş">
        <button type="submit">Gönder</button>
    </form>
    <p id="disclaimer">
Burada verilen yanıtlar yalnızca bilgi vermek amaçlıdır. Resmi veya hukuksal sorumluluk taşımamaktadır.
    </p>

    <script>
        const chatContainer = document.getElementById('chat-container');
        const messageForm = document.getElementById('message-form');
        const messageInput = document.getElementById('message-input');
        const ws = new WebSocket("ws://localhost:8000/ws");

        ws.onmessage = function(event) {
            addMessage(event.data, false);
        };

        messageForm.addEventListener('submit', function(event) {
            event.preventDefault();
            const message = messageInput.value.trim();
            if (message) {
                ws.send(message);
                addMessage(message, true);
                messageInput.value = '';
            }
        });

        function addMessage(text, isUser) {
            const messageElement = document.createElement('div');
            messageElement.classList.add('message');
            messageElement.classList.add(isUser ? 'user-message' : 'bot-message');
            messageElement.textContent = text;
            chatContainer.appendChild(messageElement);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
    </script>
</body>
</html>






"""


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)
manager = ConnectionManager()



@app.get("/")
async def get():
    return HTMLResponse(html)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    chat_history = []  # Collect chat history here (a sequence of messages)

    try:
        while True:
            client_prompt = await websocket.receive_text()
            result = rag_chain.invoke({"input": client_prompt, "chat_history": chat_history})
            # await manager.send_personal_message(f"My Prompt: {client_prompt}", websocket)
            await manager.send_personal_message(f"{result['answer']}", websocket)
            # await manager.broadcast(f"Client says: {data}")
            chat_history.append(HumanMessage(content=client_prompt))
            chat_history.append(SystemMessage(content=result["answer"]))

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast(f"Client left the chat")
