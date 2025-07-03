import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent import WebQueryAgent

app = FastAPI()

# Allow frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    result: str

agent = WebQueryAgent()

@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.post("/api/query")
async def query_endpoint(req: QueryRequest):
    # Always run process_query in a thread to support async Playwright
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, agent.process_query, req.query)
    # result is now a dict with "result" and "sources"
    return result
