from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Request
from pydantic import BaseModel
from agent import WebQueryAgent
from fastapi.responses import JSONResponse
import asyncio

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
    try:
        # Set a timeout for the agent's response (e.g., 30 seconds)
        result = agent.process_query(req.query)
        if hasattr(result, "__await__"):
            # If process_query is async, await it with timeout
            result = await asyncio.wait_for(result, timeout=30)
        else:
            # If process_query is sync, run it in a thread with timeout
            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(loop.run_in_executor(None, agent.process_query, req.query), timeout=30)
        return {"result": result}
    except asyncio.TimeoutError:
        return JSONResponse(
            status_code=504,
            content={"result": "❌ Error: Query timed out. Please try again with a simpler question."}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"result": f"❌ Error: {str(e)}"}
        )

@app.get("/api/health")
async def health_check():
    return {"status": "ok"}
