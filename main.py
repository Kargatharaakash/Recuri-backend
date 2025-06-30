from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent import WebQueryAgent

app = FastAPI()

# Allow frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://recuri-frontend.vercel.app", 
    ],
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

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)

