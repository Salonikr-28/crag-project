import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("crag_api")

# Import the LangGraph application compiled workflow from crag_app
try:
    from crag_app import app as crag_app
except ImportError as e:
    logger.error("Failed to import app from crag_app.py. Ensure crag_app.py is in the same directory.")
    raise e

app = FastAPI(
    title="CRAG Backend API",
    description="FastAPI backend for Corrective Retrieval-Augmented Generation (CRAG) system.",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskRequest(BaseModel):
    query: str = Field(..., description="The user query/question to ask the CRAG application", examples=["What is the return policy?"])

class AskResponse(BaseModel):
    query: str = Field(..., description="The original query")
    answer: str = Field(..., description="The generated answer from the CRAG application")
    docs: Optional[List[str]] = Field(default=None, description="Initially retrieved documents")
    relevant_docs: Optional[List[str]] = Field(default=None, description="Documents graded as relevant")
    final_context: Optional[List[str]] = Field(default=None, description="Final context documents used to answer the query")

@app.post("/ask", response_model=AskResponse)
async def ask_endpoint(request: AskRequest):
    """
    Exposes the CRAG workflow as a POST endpoint.
    Takes a query, runs it through retrieve, grade, web search (if needed), and generate steps.
    """
    logger.info(f"Received query: {request.query}")
    try:
        # Invoke the compiled LangGraph workflow asynchronously
        # We pass the query in the expected input schema
        result = await crag_app.ainvoke({"query": request.query})
        
        # Log success and return the response
        logger.info(f"Successfully processed query: {request.query}")
        return AskResponse(
            query=result.get("query"),
            answer=result.get("answer"),
            docs=result.get("docs"),
            relevant_docs=result.get("relevant_docs"),
            final_context=result.get("final_context")
        )
    except Exception as e:
        logger.error(f"Error processing query '{request.query}': {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while executing the CRAG workflow: {str(e)}"
        )

@app.get("/")
async def root():
    return {"message": "CRAG API is running. Send POST requests to /ask."}
