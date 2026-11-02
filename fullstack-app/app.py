"""
Workload #4: FastAPI + PostgreSQL + Redis all together
Test memory with all 3 services running simultaneously
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
import redis
import os
from datetime import datetime
import time

app = FastAPI(title="Stress Test API")

# Database
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/appdb")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Redis
redis_client = redis.Redis(host=os.getenv("REDIS_HOST", "localhost"), port=6379, decode_responses=True)

# Model
class Paper(Base):
    __tablename__ = "papers"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    author = Column(String)
    year = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# Pydantic schemas
class PaperCreate(BaseModel):
    title: str
    author: str
    year: int

class PaperResponse(BaseModel):
    id: int
    title: str
    author: str
    year: int

# Endpoints
@app.get("/")
def root():
    return {"message": "Stress Test API", "timestamp": datetime.now().isoformat()}

@app.post("/papers", response_model=PaperResponse)
def create_paper(paper: PaperCreate):
    db = SessionLocal()
    db_paper = Paper(**paper.model_dump())
    db.add(db_paper)
    db.commit()
    db.refresh(db_paper)
    db.close()

    # Cache in Redis
    redis_client.setex(f"paper:{db_paper.id}", 3600, db_paper.title)

    return db_paper

@app.get("/papers/{paper_id}")
def get_paper(paper_id: int):
    # Check cache first
    cached = redis_client.get(f"paper:{paper_id}")
    if cached:
        return {"id": paper_id, "title": cached, "cached": True}

    db = SessionLocal()
    paper = db.query(Paper).filter(Paper.id == paper_id).first()
    db.close()

    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")

    # Cache for next time
    redis_client.setex(f"paper:{paper_id}", 3600, paper.title)
    return {"id": paper.id, "title": paper.title, "cached": False}

@app.get("/stats")
def stats():
    """Check memory usage of all services"""
    import psutil
    mem = psutil.virtual_memory()
    return {
        "total_gb": mem.total / 1e9,
        "available_gb": mem.available / 1e9,
        "used_gb": mem.used / 1e9,
        "percent_used": mem.percent
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)