"""MongoDB setup and models for DASS21 scores and knowledge base."""
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

# MongoDB connection
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
DATABASE_NAME = os.getenv("DATABASE_NAME", "vietmind_ai")

# Async client for FastAPI
motor_client = AsyncIOMotorClient(MONGODB_URL)
async_db = motor_client[DATABASE_NAME]

# Sync client for non-async operations
sync_client = MongoClient(MONGODB_URL)
sync_db = sync_client[DATABASE_NAME]


class DASS21ScoreModel(BaseModel):
    """Model for DASS21 assessment scores."""
    user_id: str
    depression_score: int
    anxiety_score: int
    stress_score: int
    total_score: int
    severity_level: str  # normal, mild, moderate, severe, extremely severe
    assessment_date: datetime = Field(default_factory=datetime.utcnow)
    notes: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user123",
                "depression_score": 12,
                "anxiety_score": 10,
                "stress_score": 15,
                "total_score": 37,
                "severity_level": "moderate",
                "assessment_date": "2025-12-17T10:00:00",
                "notes": "Initial assessment"
            }
        }


class KnowledgeBaseModel(BaseModel):
    """Model for knowledge base documents."""
    title: str
    content: str
    category: str
    source: str
    embedding: Optional[List[float]] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "title": "Understanding Depression",
                "content": "Depression is a common mental health condition...",
                "category": "mental_health",
                "source": "WHO Guidelines",
                "created_at": "2025-12-17T10:00:00"
            }
        }


class ChatSessionModel(BaseModel):
    """Model for chat sessions."""
    session_id: str
    user_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_activity: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = True

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "sess_123456",
                "user_id": "user123",
                "created_at": "2025-12-17T10:00:00",
                "last_activity": "2025-12-17T10:30:00",
                "is_active": True
            }
        }


class ChatMessageModel(BaseModel):
    """Model for individual chat messages."""
    session_id: str
    user_id: str
    role: str  # "user" or "assistant"
    content: str
    query_type: Optional[str] = None  # "dass21_query", "knowledge_search", "general"
    metadata: Optional[dict] = None  # Additional context like tool results
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "sess_123456",
                "user_id": "user123",
                "role": "user",
                "content": "How are my DASS21 scores?",
                "query_type": "dass21_query",
                "timestamp": "2025-12-17T10:00:00"
            }
        }


async def init_db():
    """Initialize MongoDB collections and indexes."""
    try:
        # Test connection first
        await motor_client.admin.command('ping')
        print(f"✓ Connected to MongoDB at {MONGODB_URL}")

        # Create indexes for DASS21 scores
        await async_db.dass21_scores.create_index("user_id")
        await async_db.dass21_scores.create_index("assessment_date")

        # Create indexes for knowledge base
        await async_db.knowledge_base.create_index("title")
        await async_db.knowledge_base.create_index("category")
        # Create text index for full-text search
        await async_db.knowledge_base.create_index([("content", "text"), ("title", "text")])

        # Create indexes for chat sessions
        await async_db.chat_sessions.create_index("session_id", unique=True)
        await async_db.chat_sessions.create_index("user_id")
        await async_db.chat_sessions.create_index("last_activity")

        # Create indexes for chat messages
        await async_db.chat_messages.create_index("session_id")
        await async_db.chat_messages.create_index("user_id")
        await async_db.chat_messages.create_index("timestamp")
        await async_db.chat_messages.create_index([("session_id", 1), ("timestamp", 1)])

        print("✓ MongoDB initialized successfully with all indexes")
    except Exception as e:
        print(f"✗ MongoDB connection failed: {e}")
        print(f"  Please ensure MongoDB is running at {MONGODB_URL}")
        print(f"  Or update MONGODB_URL in .env file")
        raise


async def get_dass21_collection():
    """Get DASS21 scores collection."""
    return async_db.dass21_scores


async def get_knowledge_base_collection():
    """Get knowledge base collection."""
    return async_db.knowledge_base


async def get_chat_sessions_collection():
    """Get chat sessions collection."""
    return async_db.chat_sessions


async def get_chat_messages_collection():
    """Get chat messages collection."""
    return async_db.chat_messages
