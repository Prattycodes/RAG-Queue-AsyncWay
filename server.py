from fastapi import FastAPI, HTTPException, Query
from redis.exceptions import ConnectionError as RedisConnectionError

from client.rq_client import queue
from queues.worker import process_query

app = FastAPI()


@app.get("/")
def root():
    return {"status": "Server is running"}


@app.post("/chat")
def chat(query: str = Query(..., description="The chat query of user")):
    try:
        job = queue.enqueue(process_query, query)
    except RedisConnectionError:
        raise HTTPException(
            status_code=503,
            detail="Redis is not running. Start it with: docker compose up -d",
        )
    return {"status": "queued", "jobid": job.id}


@app.get("/job-status")
def get_result(jobid: str = Query(..., description="JOB ID")):
    job = queue.fetch_job(jobid)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.is_finished:
        return {"status": "finished", "result": job.return_value()}
    if job.is_failed:
        return {"status": "failed", "error": job.exc_info}
    return {"status": job.get_status()}