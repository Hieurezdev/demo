"""Short-term memory management for chat sessions using MongoDB."""
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import uuid
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from backend.database import async_db, ChatSessionModel, ChatMessageModel


class ChatMemoryManager:
    """Manages chat history and short-term memory with MongoDB."""

    def __init__(self, max_history: int = 10, session_timeout_minutes: int = 30):
        """
        Initialize memory manager.

        Args:
            max_history: Maximum number of messages to keep in short-term memory
            session_timeout_minutes: Minutes of inactivity before session expires
        """
        self.max_history = max_history
        self.session_timeout = timedelta(minutes=session_timeout_minutes)

    async def create_session(self, user_id: str) -> str:
        """
        Create a new chat session.

        Args:
            user_id: User identifier

        Returns:
            session_id: Unique session identifier
        """
        session_id = f"sess_{uuid.uuid4().hex[:12]}"

        session = ChatSessionModel(
            session_id=session_id,
            user_id=user_id
        )

        await async_db.chat_sessions.insert_one(session.dict())

        return session_id

    async def get_or_create_session(self, user_id: str, session_id: Optional[str] = None) -> str:
        """
        Get existing active session or create a new one.

        Args:
            user_id: User identifier
            session_id: Optional existing session ID

        Returns:
            session_id: Active session identifier
        """
        if session_id:
            # Check if session exists and is active
            session = await async_db.chat_sessions.find_one({"session_id": session_id})

            if session and session.get("is_active"):
                # Check if session hasn't expired
                last_activity = session.get("last_activity")
                if datetime.utcnow() - last_activity < self.session_timeout:
                    # Update last activity
                    await async_db.chat_sessions.update_one(
                        {"session_id": session_id},
                        {"$set": {"last_activity": datetime.utcnow()}}
                    )
                    return session_id

        # Create new session if none exists or expired
        return await self.create_session(user_id)

    async def add_message(
        self,
        session_id: str,
        user_id: str,
        role: str,
        content: str,
        query_type: Optional[str] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Add a message to the chat history.

        Args:
            session_id: Session identifier
            user_id: User identifier
            role: Message role ("user" or "assistant")
            content: Message content
            query_type: Type of query (optional)
            metadata: Additional metadata (optional)
        """
        message = ChatMessageModel(
            session_id=session_id,
            user_id=user_id,
            role=role,
            content=content,
            query_type=query_type,
            metadata=metadata
        )

        await async_db.chat_messages.insert_one(message.dict())

        # Update session last activity
        await async_db.chat_sessions.update_one(
            {"session_id": session_id},
            {"$set": {"last_activity": datetime.utcnow()}}
        )

    async def get_recent_messages(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """
        Get recent messages from the session (short-term memory).

        Args:
            session_id: Session identifier
            limit: Maximum number of messages to retrieve (defaults to max_history)

        Returns:
            List of messages sorted by timestamp
        """
        if limit is None:
            limit = self.max_history

        cursor = async_db.chat_messages.find(
            {"session_id": session_id}
        ).sort("timestamp", -1).limit(limit)

        messages = await cursor.to_list(length=limit)

        # Reverse to get chronological order
        messages.reverse()

        # Convert ObjectId to string
        for msg in messages:
            msg['_id'] = str(msg['_id'])
            msg['timestamp'] = msg['timestamp'].isoformat() if isinstance(msg['timestamp'], datetime) else msg['timestamp']

        return messages

    async def get_conversation_history(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[BaseMessage]:
        """
        Get conversation history as LangChain messages.

        Args:
            session_id: Session identifier
            limit: Maximum number of messages to retrieve

        Returns:
            List of LangChain BaseMessage objects
        """
        messages = await self.get_recent_messages(session_id, limit)

        langchain_messages = []
        for msg in messages:
            if msg["role"] == "user":
                langchain_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                langchain_messages.append(AIMessage(content=msg["content"]))

        return langchain_messages

    async def get_session_history(self, user_id: str, limit: int = 10) -> List[Dict]:
        """
        Get all sessions for a user.

        Args:
            user_id: User identifier
            limit: Maximum number of sessions to retrieve

        Returns:
            List of session objects
        """
        cursor = async_db.chat_sessions.find(
            {"user_id": user_id}
        ).sort("last_activity", -1).limit(limit)

        sessions = await cursor.to_list(length=limit)

        # Convert ObjectId to string
        for session in sessions:
            session['_id'] = str(session['_id'])

        return sessions

    async def get_full_session_messages(self, session_id: str) -> List[Dict]:
        """
        Get all messages from a session (not limited by max_history).

        Args:
            session_id: Session identifier

        Returns:
            List of all messages in the session
        """
        cursor = async_db.chat_messages.find(
            {"session_id": session_id}
        ).sort("timestamp", 1)

        messages = await cursor.to_list(length=None)

        # Convert ObjectId to string
        for msg in messages:
            msg['_id'] = str(msg['_id'])
            msg['timestamp'] = msg['timestamp'].isoformat() if isinstance(msg['timestamp'], datetime) else msg['timestamp']

        return messages

    async def end_session(self, session_id: str):
        """
        Mark a session as inactive.

        Args:
            session_id: Session identifier
        """
        await async_db.chat_sessions.update_one(
            {"session_id": session_id},
            {"$set": {"is_active": False}}
        )

    async def delete_session(self, session_id: str):
        """
        Delete a session and all its messages.

        Args:
            session_id: Session identifier
        """
        # Delete all messages
        await async_db.chat_messages.delete_many({"session_id": session_id})

        # Delete session
        await async_db.chat_sessions.delete_one({"session_id": session_id})

    async def cleanup_old_sessions(self, days: int = 7):
        """
        Clean up sessions older than specified days.

        Args:
            days: Number of days to keep sessions
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)

        # Find old sessions
        cursor = async_db.chat_sessions.find(
            {"last_activity": {"$lt": cutoff_date}}
        )

        old_sessions = await cursor.to_list(length=None)

        # Delete messages and sessions
        for session in old_sessions:
            await self.delete_session(session["session_id"])

        return len(old_sessions)


# Global memory manager instance
memory_manager = ChatMemoryManager(max_history=10, session_timeout_minutes=30)
