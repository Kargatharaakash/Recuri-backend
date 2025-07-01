from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from pydantic import BaseModel
from agent import WebQueryAgent
from fastapi.responses import JSONResponse
import asyncio

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://recuri-frontend.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    result: str

agent = WebQueryAgent()

@app.post("/api/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    try:
        loop = asyncio.get_event_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(None, agent.process_query, req.query), timeout=30
        )
        return {"result": result}
    except asyncio.TimeoutError:
        return JSONResponse(
            status_code=504,
            content={"result": "❌ Timeout. Try again."}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"result": f"❌ Error: {str(e)}"}
        )

@app.get("/api/health")
async def health_check():
    return {"status": "ok"}
