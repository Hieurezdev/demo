"""FastAPI backend for VietMind AI RAG system with chat history."""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

from backend.database import init_db, async_db, DASS21ScoreModel, KnowledgeBaseModel
from backend.agent import process_message
from backend.memory import memory_manager

app = FastAPI(
    title="VietMind AI - Agentic RAG API",
    description="API for DASS21 assessment and mental health knowledge search with conversation memory",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = ""
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    user_id: str
    session_id: str
    query_type: str
    timestamp: datetime


class DASS21CreateRequest(BaseModel):
    user_id: str
    depression_score: int
    anxiety_score: int
    stress_score: int
    notes: Optional[str] = None


class KnowledgeCreateRequest(BaseModel):
    title: str
    content: str
    category: str
    source: str


@app.on_event("startup")
async def startup_event():
    """Initialize database on startup."""
    await init_db()
    print("Application started successfully")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "VietMind AI - Agentic RAG API with Conversation Memory",
        "version": "1.0.0",
        "status": "running",
        "features": [
            "DASS21 score tracking",
            "Parallel knowledge search (Google + MongoDB)",
            "Conversation history and short-term memory",
            "LangGraph agent routing"
        ]
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint - processes user messages through the LangGraph agent with conversation memory.

    The agent will automatically route to:
    - DASS21 score queries for mental health assessments
    - Parallel knowledge search (Google + MongoDB) for information queries
    - General conversation for other interactions

    Conversation history is maintained per session for context-aware responses.
    """
    try:
        print(f"📩 Received message from user: {request.user_id}")
        print(f"📝 Message: {request.message[:50]}...")

        response, session_id, query_type = await process_message(
            message=request.message,
            user_id=request.user_id if request.user_id else "default_user",
            session_id=request.session_id
        )

        print(f"✅ Response generated successfully")

        return ChatResponse(
            response=response,
            user_id=request.user_id if request.user_id else "default_user",
            session_id=session_id,
            query_type=query_type,
            timestamp=datetime.utcnow()
        )
    except Exception as e:
        import traceback
        print(f"❌ Error processing message:")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")


@app.get("/chat/history/{session_id}")
async def get_chat_history(session_id: str, limit: Optional[int] = None):
    """
    Get chat history for a specific session.

    Args:
        session_id: Session identifier
        limit: Optional limit on number of messages (defaults to last 10 for short-term memory)
    """
    try:
        if limit:
            messages = await memory_manager.get_recent_messages(session_id, limit)
        else:
            messages = await memory_manager.get_full_session_messages(session_id)

        return {
            "session_id": session_id,
            "messages": messages,
            "count": len(messages)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching chat history: {str(e)}")


@app.get("/chat/sessions/{user_id}")
async def get_user_sessions(user_id: str, limit: int = 10):
    """
    Get all chat sessions for a user.

    Args:
        user_id: User identifier
        limit: Maximum number of sessions to return
    """
    try:
        sessions = await memory_manager.get_session_history(user_id, limit)

        return {
            "user_id": user_id,
            "sessions": sessions,
            "count": len(sessions)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching user sessions: {str(e)}")


@app.post("/chat/session/new")
async def create_new_session(user_id: str):
    """
    Create a new chat session for a user.

    Args:
        user_id: User identifier
    """
    try:
        session_id = await memory_manager.create_session(user_id)

        return {
            "status": "success",
            "session_id": session_id,
            "user_id": user_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating session: {str(e)}")


@app.delete("/chat/session/{session_id}")
async def delete_session(session_id: str):
    """
    Delete a chat session and all its messages.

    Args:
        session_id: Session identifier
    """
    try:
        await memory_manager.delete_session(session_id)

        return {
            "status": "success",
            "message": f"Session {session_id} deleted successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting session: {str(e)}")


@app.post("/chat/session/{session_id}/end")
async def end_session(session_id: str):
    """
    Mark a chat session as inactive (end the session).

    Args:
        session_id: Session identifier
    """
    try:
        await memory_manager.end_session(session_id)

        return {
            "status": "success",
            "message": f"Session {session_id} ended successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error ending session: {str(e)}")


@app.post("/dass21/create")
async def create_dass21_score(request: DASS21CreateRequest):
    """Create a new DASS21 assessment score."""
    try:
        # Calculate total score
        total_score = request.depression_score + request.anxiety_score + request.stress_score

        # Determine severity level (simplified categorization)
        if total_score <= 20:
            severity = "normal"
        elif total_score <= 40:
            severity = "mild"
        elif total_score <= 60:
            severity = "moderate"
        elif total_score <= 80:
            severity = "severe"
        else:
            severity = "extremely severe"

        score_data = DASS21ScoreModel(
            user_id=request.user_id,
            depression_score=request.depression_score,
            anxiety_score=request.anxiety_score,
            stress_score=request.stress_score,
            total_score=total_score,
            severity_level=severity,
            notes=request.notes
        )

        # Insert into MongoDB
        collection = async_db.dass21_scores
        result = await collection.insert_one(score_data.dict())

        return {
            "status": "success",
            "id": str(result.inserted_id),
            "total_score": total_score,
            "severity_level": severity
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating DASS21 score: {str(e)}")


@app.get("/dass21/{user_id}")
async def get_dass21_scores(user_id: str, limit: int = 10):
    """Get DASS21 scores for a user."""
    try:
        collection = async_db.dass21_scores
        cursor = collection.find({"user_id": user_id}).sort("assessment_date", -1).limit(limit)
        scores = await cursor.to_list(length=limit)

        # Convert ObjectId to string
        for score in scores:
            score['_id'] = str(score['_id'])

        return {
            "user_id": user_id,
            "scores": scores,
            "count": len(scores)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching DASS21 scores: {str(e)}")


@app.post("/knowledge/create")
async def create_knowledge(request: KnowledgeCreateRequest):
    """Add a new document to the knowledge base."""
    try:
        knowledge_data = KnowledgeBaseModel(
            title=request.title,
            content=request.content,
            category=request.category,
            source=request.source
        )

        collection = async_db.knowledge_base
        result = await collection.insert_one(knowledge_data.dict())

        return {
            "status": "success",
            "id": str(result.inserted_id)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating knowledge base entry: {str(e)}")


@app.get("/knowledge/search")
async def search_knowledge(query: str, category: Optional[str] = None, limit: int = 5):
    """Search the knowledge base."""
    try:
        collection = async_db.knowledge_base

        # Build query
        search_query = {"$text": {"$search": query}}
        if category:
            search_query["category"] = category

        # Execute search
        cursor = collection.find(
            search_query,
            {"score": {"$meta": "textScore"}}
        ).sort([("score", {"$meta": "textScore"})]).limit(limit)

        results = await cursor.to_list(length=limit)

        # Convert ObjectId to string
        for result in results:
            result['_id'] = str(result['_id'])

        return {
            "query": query,
            "results": results,
            "count": len(results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching knowledge base: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Test MongoDB connection
        await async_db.command("ping")
        return {
            "status": "healthy",
            "database": "connected",
            "timestamp": datetime.utcnow()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "database": "disconnected",
            "error": str(e),
            "timestamp": datetime.utcnow()
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
