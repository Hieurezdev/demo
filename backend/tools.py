"""Tools for the LangGraph agent."""
from langchain.tools import tool
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from googleapiclient.discovery import build
import os
from google import genai
from dotenv import load_dotenv
load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
def get_embedding(text: str) -> list[float]:
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents= text)
    return result.embeddings[0].values

@tool
def query_dass21_scores(user_id: str, days: int = 30) -> Dict:
    """
    Query DASS21 scores from MongoDB database for a specific user.

    Args:
        user_id: The user ID to query scores for
        days: Number of days to look back (default 30)

    Returns:
        Dictionary containing user's DASS21 scores and history
    """
    from backend.database import sync_db

    # Calculate date range
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)

    # Query MongoDB
    collection = sync_db.dass21_scores
    scores = list(collection.find({
        "user_id": user_id,
        "assessment_date": {"$gte": start_date, "$lte": end_date}
    }).sort("assessment_date", -1))

    if not scores:
        return {
            "status": "no_data",
            "message": f"No DASS21 scores found for user {user_id} in the last {days} days",
            "user_id": user_id
        }

    # Get latest score
    latest = scores[0]

    # Convert ObjectId to string for JSON serialization
    for score in scores:
        score['_id'] = str(score['_id'])
        score['assessment_date'] = score['assessment_date'].isoformat()

    return {
        "status": "success",
        "user_id": user_id,
        "latest_score": {
            "depression": latest.get("depression_score"),
            "anxiety": latest.get("anxiety_score"),
            "stress": latest.get("stress_score"),
            "total": latest.get("total_score"),
            "severity": latest.get("severity_level"),
            "date": latest.get("assessment_date")
        },
        "history": scores,
        "total_assessments": len(scores)
    }


@tool
def search_knowledge_base(query: str, category: Optional[str] = None, vector_search_index: str = "default", candidates: int = 10, limit: int = 5) -> List[Dict]:
    """
    Search knowledge base in MongoDB for relevant information.

    Args:
        query: Search query string
        category: Optional category filter (e.g., "mental_health", "coping_strategies")
        limit: Maximum number of results to return (default 5)

    Returns:
        List of relevant documents from knowledge base
    """
    from backend.database import sync_db
    collection = sync_db.knowledge_base

    try:
        query_vector = get_embedding(query)
        
        search_params = {
            "index": vector_search_index,
            "path": "embedding",
            "queryVector": query_vector,
            "limit": limit,
            "numCandidates": candidates
        }

        if category:
            search_params["filter"] = {"category": {"$eq": category}}

        vector_pipeline = [{"$vectorSearch": search_params}]
        vector_results = list(collection.aggregate(vector_pipeline))
        
        for result in vector_results:
            result['_id'] = str(result['_id'])
            for date_field in ['created_at', 'updated_at']:
                if date_field in result and result[date_field]:
                    result[date_field] = result[date_field].isoformat()
                    
        return vector_results

    except Exception as e:
        print(f"Error searching knowledge base: {e}")
        return []

        return vector_results
    except Exception as e:
        # If text index doesn't exist or knowledge base is empty, return empty results
        print(f"Warning: Knowledge base search failed: {e}")
        return []
    
@tool
def google_search(query: str, num_results: int = 5) -> List[Dict]:
    """
    Search Google for relevant information using Google Custom Search API.

    Args:
        query: Search query string
        num_results: Number of results to return (default 5)

    Returns:
        List of search results with title, link, and snippet
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")

    if not api_key or not search_engine_id:
        return [{
            "error": "Google API key or Search Engine ID not configured",
            "message": "Please set GOOGLE_API_KEY and GOOGLE_SEARCH_ENGINE_ID environment variables"
        }]

    try:
        service = build("customsearch", "v1", developerKey=api_key)
        result = service.cse().list(
            q=query,
            cx=search_engine_id,
            num=num_results
        ).execute()

        items = result.get("items", [])

        return [
            {
                "title": item.get("title"),
                "link": item.get("link"),
                "snippet": item.get("snippet"),
                "source": "google_search"
            }
            for item in items
        ]
    except Exception as e:
        return [{
            "error": "Google search failed",
            "message": str(e)
        }]


@tool
def parallel_knowledge_search(query: str, category: Optional[str] = None) -> Dict:
    """
    Search both Google and knowledge base in parallel for comprehensive results.
    This tool is used when user asks about knowledge or information.

    Args:
        query: Search query string
        category: Optional category filter for knowledge base

    Returns:
        Combined results from both Google and knowledge base
    """
    # Execute both searches
    kb_results = search_knowledge_base.invoke({"query": query, "category": category})
    google_results = google_search.invoke({"query": query})

    return {
        "query": query,
        "knowledge_base_results": kb_results,
        "google_results": google_results,
        "total_results": len(kb_results) + len(google_results)
    }


# Export all tools
ALL_TOOLS = [
    query_dass21_scores,
    search_knowledge_base,
    google_search,
    parallel_knowledge_search
]
