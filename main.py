from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent import WebQueryAgent
from fastapi.responses import JSONResponse

app = FastAPI()

# Allow frontend dev server and production frontend
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

@app.options("/{rest_of_path:path}")
async def preflight_handler(rest_of_path: str):
    """
    Handle CORS preflight requests for any path.
    """
    return JSONResponse(content={}, status_code=204)

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

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
