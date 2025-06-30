from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Request
from pydantic import BaseModel
from agent import WebQueryAgent
from fastapi.responses import JSONResponse

app = FastAPI()

# CORS middleware must be added before any routes
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

@app.options("/{rest_of_path:path}")
async def preflight(rest_of_path: str):
    # Explicitly handle CORS preflight requests
    return JSONResponse(status_code=204)

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
