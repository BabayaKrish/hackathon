# audit.py
import asyncio
import json
from google.cloud import bigquery
from datetime import datetime
import uuid
import os

client = bigquery.Client()
TABLE_ID = os.getenv("BQ_TABLE_ID", "ccibt-hack25ww7-743.AuditTrail.AgentEvents")

async def async_log_event(
    agent_name,
    model_name,
    caller,
    input_text,
    output_text,
    latency_ms,
    tokens_used=None,
    safety_result="allowed",
    error_message=None,
    metadata=None
):
    """Log agent execution to audit trail"""
    row = {
        "event_id": str(uuid.uuid4()),
        "event_timestamp": datetime.utcnow().isoformat(),
        "agent_name": agent_name,
        "model_name": model_name,
        "caller": caller,
        "input_text": input_text,
        "output_text": output_text,
        "safety_result": safety_result,
        "error_message": error_message,
        "latency_ms": latency_ms,
        "tokens_used": tokens_used,
        "metadata": json.dumps(metadata or {})
    }

    try:
        # Run BigQuery insert in thread pool to avoid blocking
        errors = await asyncio.to_thread(
            client.insert_rows_json, 
            TABLE_ID, 
            [row]
        )
        
        if errors:
            print(f"Audit log errors: {errors}")
            return {"success": False, "error": errors}
        else:
            print(f"Audit event logged: {row['event_id']}")
            return {"success": True, "event_id": row['event_id']}
    except Exception as e:
        print(f"Failed to log audit event: {e}")
        return {"success": False, "error": str(e)}
