from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent import WebQueryAgent

app = FastAPI()

# Allow frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    result: str

agent = WebQueryAgent()

@app.post("/api/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    result = agent.process_query(req.query)
    if hasattr(result, "__await__"):
        # If process_query is async, await it
        result = await result
    return {"result": result}

@app.get("/api/health")
async def health_check():
    return {"status": "ok"}
