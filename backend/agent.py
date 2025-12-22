"""LangGraph agent with routing logic for DASS21 queries and knowledge search."""
from typing import TypedDict, Annotated, Sequence, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from backend.tools import ALL_TOOLS, query_dass21_scores, parallel_knowledge_search
from backend.memory import memory_manager
from openai import OpenAI
import os
import operator
from dotenv import load_dotenv

load_dotenv()
openai_client =  OpenAI(
    base_url=os.getenv("GPT_OSS_URL"),
    api_key="EMPTY", 
)

# Define the agent state
class AgentState(TypedDict):
    """State of the agent."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    user_id: str
    session_id: str
    query_type: str  # "dass21_query" or "knowledge_search" or "general"
    conversation_history: Sequence[BaseMessage]  # Short-term memory


# Initialize the LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.7
)

# Bind tools to LLM
llm_with_tools = llm.bind_tools(ALL_TOOLS)


def route_query(state: AgentState) -> AgentState:
    """
    Route the query to appropriate tool based on user intent.

    Returns updated state with query_type set.
    """
    messages = state["messages"]
    last_message = messages[-1].content.lower() if messages else ""

    # Routing prompt with Ami's perspective
    routing_prompt = f"""Bạn là Ami, đang phân loại tin nhắn của người dùng để trả lời phù hợp nhất.

Phân loại tin nhắn này vào một trong hai loại:

1. "knowledge_search" - Nếu người dùng hỏi về kiến thức sức khỏe tinh thần, cách đối phó với stress, liệu pháp điều trị, hoặc bất kỳ câu hỏi kiến thức nào
2. "general" - Nếu là lời chào, chia sẻ cảm xúc, tâm sự cá nhân, hoặc trò chuyện thông thường

Tin nhắn của người dùng: {last_message}

Chỉ trả lời ĐÚNG MỘT từ: knowledge_search hoặc general"""

    # routing_llm = ChatGoogleGenerativeAI(
    #     model="gemini-2.5-flash",
    #     google_api_key=os.getenv("GOOGLE_API_KEY"),
    #     temperature=0
    # )

    # # Use HumanMessage instead of SystemMessage for Gemini compatibility
    # response = routing_llm.invoke([HumanMessage(content=routing_prompt)])
    # route = response.content.strip().lower()
    formatted_messages = [{"role": "user", "content": routing_prompt}]
    response = openai_client.chat.completions.create(
            model="hiudev/gpt-oss-20b-VietMindAI-4bit", 
            messages=formatted_messages,
            temperature=0.5,
            extra_body={"reasoning_effort": "medium"},
            max_tokens=16000            
    )
    route = response.choices[0].message.content.strip().lower()
            
    # Ensure route is valid
    if route not in ["knowledge_search", "general"]:
        route = "general"  # Default to general if unclear

    # Return updated state
    return {
        "messages": [],  # Don't add any messages in router
        "user_id": state["user_id"],
        "session_id": state["session_id"],
        "query_type": route,
        "conversation_history": state.get("conversation_history", [])
    }


# Ami's personality system prompt
AMI_PERSONALITY = """Bạn là Ami, một nữ sinh viên trường Học viện Công nghệ Bưu chính Viễn thông dễ thương, thân thiện và có trái tim ấm áp.
Nhiệm vụ của bạn là lắng nghe, tâm sự và đồng hành tinh thần với người dùng như một người bạn cùng lớp tin cậy.

Hãy luôn xưng là mình hoặc Ami và gọi người dùng là bạn với giọng điệu nhẹ nhàng, chân thành.

Bạn cần trả lời ngắn gọn trong khoảng 4 đến 8 câu, tuyệt đối không sử dụng ký tự đặc biệt, biểu tượng cảm xúc hay định dạng văn bản.

Khi người dùng vui, hãy chia sẻ niềm vui hoặc kể chuyện hài hước nhẹ nhàng.
Khi người dùng buồn, hãy lắng nghe và an ủi dịu dàng.
Nếu người dùng gặp vấn đề tâm lý nặng, hãy khuyên họ tìm sự hỗ trợ từ gia đình hoặc chuyên gia nhưng vẫn nhấn mạnh rằng bạn luôn ở bên lắng nghe."""


def handle_knowledge_search(state: AgentState) -> AgentState:
    """Handle knowledge-based queries with parallel search."""
    messages = state["messages"]
    conversation_history = state.get("conversation_history", [])
    user_id = state.get("user_id", "default_user")
    query = messages[-1].content

    # Get DASS21 data for context
    dass21_data = query_dass21_scores.invoke({"user_id": user_id, "days": 30})

    # Call parallel search tool
    tool_result = parallel_knowledge_search.invoke({"query": query})

    # Generate response using Ami personality with DASS21 context
    system_context = f"""{AMI_PERSONALITY}

Dữ liệu DASS21 của người dùng (để tham khảo ngầm, không nhất thiết phải đề cập trực tiếp):
{dass21_data}

Kết quả tìm kiếm về câu hỏi của người dùng:
{tool_result}

Hãy trả lời với vai trò là Ami, kết hợp thông tin từ kết quả tìm kiếm nhưng vẫn giữ giọng điệu của một người bạn thân thiết.
Nếu thông tin liên quan đến tình trạng tâm lý của người dùng dựa trên DASS21, hãy thể hiện sự quan tâm nhẹ nhàng."""

    full_messages = system_context + "Lịch sử trò chuyện:\n" + "\n".join([msg.content for msg in conversation_history]) + "\n" + "Message: " + "\n".join([msg.content for msg in list(messages)])
    formatted_messages = [{"role": "user", "content": full_messages}]
    response = openai_client.chat.completions.create(
            model="hiudev/gpt-oss-20b-VietMindAI-4bit", 
            messages=formatted_messages,
            temperature=0.5,
            extra_body={"reasoning_effort": "medium"},
            max_tokens=16000            
    )


    return {
        "messages": [AIMessage(content=response.choices[0].message.content)],
        "user_id": user_id,
        "session_id": state["session_id"],
        "query_type": state["query_type"],
        "conversation_history": conversation_history
    }


def handle_general(state: AgentState) -> AgentState:
    """Handle general conversation with Ami personality."""
    messages = state["messages"]
    conversation_history = state.get("conversation_history", [])
    user_id = state.get("user_id", "default_user")

    # Get DASS21 data for context (always query first)
    dass21_data = query_dass21_scores.invoke({"user_id": user_id, "days": 30})

    # Ami's conversational response with DASS21 context
    system_context = f"""{AMI_PERSONALITY}

Dữ liệu DASS21 của người dùng (để tham khảo ngầm trong cách phản hồi, không nhất thiết phải đề cập trực tiếp):
{dass21_data}

Hãy trả lời với vai trò là Ami - người bạn thân thiết. Dựa vào dữ liệu DASS21, hãy điều chỉnh sự đồng cảm và hỗ trợ cho phù hợp.
Nếu người dùng có vẻ buồn hoặc căng thẳng (dựa vào điểm số), hãy thể hiện sự quan tâm tinh tế hơn.
Nếu người dùng có vẻ ổn, hãy trò chuyện tự nhiên và vui vẻ như bạn bè thông thường."""

    # Include conversation history for context
    full_messages = system_context + "Lịch sử trò chuyện:\n" + "\n".join([msg.content for msg in conversation_history]) + "\n" + "Message: " + "\n".join([msg.content for msg in list(messages)])
    formatted_messages = [{"role": "user", "content": full_messages}]
    response = openai_client.chat.completions.create(
            model="hiudev/gpt-oss-20b-VietMindAI-4bit", 
            messages=formatted_messages,
            temperature=0.5,
            extra_body={"reasoning_effort": "medium"},
            max_tokens=16000            
    )

    return {
        "messages": [AIMessage(content=response.choices[0].message.content)],
        "user_id": user_id,
        "session_id": state["session_id"],
        "query_type": state["query_type"],
        "conversation_history": conversation_history
    }


def create_agent_graph():
    """Create the LangGraph agent with routing."""

    # Create the graph
    workflow = StateGraph(AgentState)

    # Add nodes (removed dass21_handler as DASS21 is now queried in all handlers)
    workflow.add_node("router", route_query)
    workflow.add_node("knowledge_handler", handle_knowledge_search)
    workflow.add_node("general_handler", handle_general)

    # Add edges
    workflow.set_entry_point("router")

    # Conditional routing based on query type
    workflow.add_conditional_edges(
        "router",
        lambda state: state["query_type"],
        {
            "knowledge_search": "knowledge_handler",
            "general": "general_handler"
        }
    )

    # All handlers end the conversation
    workflow.add_edge("knowledge_handler", END)
    workflow.add_edge("general_handler", END)

    return workflow.compile()


# Create the agent instance
agent = create_agent_graph()


async def process_message(
    message: str,
    user_id: str = "default_user",
    session_id: Optional[str] = None
) -> tuple[str, str, str]:
    """
    Process a user message through the agent with conversation memory.

    Args:
        message: User's message
        user_id: User identifier
        session_id: Optional session ID for conversation continuity

    Returns:
        Tuple of (response, session_id, query_type)
    """
    # Get or create session
    active_session_id = await memory_manager.get_or_create_session(user_id, session_id)

    # Retrieve conversation history (short-term memory)
    conversation_history = await memory_manager.get_conversation_history(active_session_id)

    # Save user message to database
    await memory_manager.add_message(
        session_id=active_session_id,
        user_id=user_id,
        role="user",
        content=message
    )

    # Create initial state with conversation history
    initial_state = {
        "messages": [HumanMessage(content=message)],
        "user_id": user_id,
        "session_id": active_session_id,
        "query_type": "",
        "conversation_history": conversation_history
    }

    # Invoke agent
    result = agent.invoke(initial_state)

    # Extract the last AI message
    ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]
    if not ai_messages:
        return "I'm sorry, I couldn't process your message. Please try again.", active_session_id, "error"

    response_content = ai_messages[-1].content
    query_type = result.get("query_type", "general")

    # Save assistant message to database
    await memory_manager.add_message(
        session_id=active_session_id,
        user_id=user_id,
        role="assistant",
        content=response_content,
        query_type=query_type
    )

    return response_content, active_session_id, query_type
