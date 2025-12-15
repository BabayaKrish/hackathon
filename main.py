"""
FastAPI Backend - ADK Sequential Agents
Streamlit calls this API → FastAPI executes agents sequentially
"""

import os
import logging
from datetime import datetime
import pandas as pd
import io
from typing import Dict, Any

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from src.services.agent_executor import SequentialAgentExecutor
from src.models.schemas import PipelineRequest, PipelineResponse

# Load environment
load_dotenv('.env.local')

# Logging
logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO'))
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="ADK Sequential Agent Pipeline",
    description="Streamlit → FastAPI → Google ADK Agents",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Agent Executor
logger.info("Initializing Agent Executor...")
agent_executor = SequentialAgentExecutor()
logger.info("✅ Agent Executor initialized")

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    return {
        "message": "ADK Sequential Agent Pipeline v2.0",
        "docs": "/docs",
        "agents": ["schema_detector", "field_mapper", "cleaner", "validator"]
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "agents": "ready"}

@app.post("/process-with-agents")
async def process_pipeline(
    file: UploadFile = File(...),
    dataset_name: str = Query("dataset")
) -> Dict[str, Any]:
    """
    Main endpoint: Upload CSV → Execute sequential agents → Return results
    
    Execution order:
    1. SchemaDetector → Analyzes structure
    2. FieldMapper → Maps fields
    3. DataCleaner → Cleans data
    4. DataValidator → Validates quality
    """
    try:
        # Read CSV file
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        csv_content = df.to_csv(index=False)
        
        logger.info(f"Processing {len(df)} rows from {file.filename}")
        
        # Execute agents sequentially
        pipeline_result = agent_executor.execute_pipeline(
            csv_content=csv_content,
            dataset_name=dataset_name
        )
        
        return pipeline_result
    
    except Exception as e:
        logger.error(f"Pipeline error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agents")
async def get_agents():
    """Get list of available agents"""
    agents = agent_executor.get_available_agents()
    return {
        "agents": agents,
        "execution_order": agents,
        "count": len(agents)
    }

if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Starting ADK Sequential Agent Pipeline...")
    logger.info(f"📍 Available agents: {agent_executor.get_available_agents()}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
