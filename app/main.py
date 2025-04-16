from fastapi import FastAPI
from pydantic import BaseModel
import httpx

app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str

@app.post("/chat")
async def chat_with_ollama(request: PromptRequest):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://ollama:11434/api/generate",
            json={
                "model": "llama3",  # غيّر الاسم حسب النموذج اللي منزل عليه
                "prompt": request.prompt,
                "stream": False
            }
        )
    data = response.json()
    return {"response": data.get("response", "")}
