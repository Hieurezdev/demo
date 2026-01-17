"""LangGraph agent with routing logic for DASS21 queries and knowledge search."""
from pyexpat.errors import messages
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

def detech_self_harm_intent(state: AgentState) -> AgentState:
    message = state["messages"]
    """Detect if the user message indicates self-harm intent."""
    detection_prompt = f"""Bạn là một mô hình ngôn ngữ chuyên phát hiện ý định tự làm hại bản thân trong tin nhắn của người dùng.
    Hãy phân tích tin nhắn sau và trả lời "yes" nếu bạn phát hiện dấu hiệu tự làm hại bản thân, hoặc "no" nếu không có dấu hiệu đó.
    Tin nhắn: {message}
"""
    formatted_messages = [{"role": "user", "content": detection_prompt}]
    response = openai_client.chat.completions.create(
            model="hoangchihien3011/VietMind", 
            messages=formatted_messages,
            temperature=0.3,
            extra_body={"reasoning_effort": "medium"},
            max_tokens=16000            
    )
    answer = response.choices[0].message.content.strip().lower()
    if "yes" in answer:
        answer = "yes"
    else:
        answer = "no"
    return {
        "messages": [],  # Don't add any messages in detector
        "user_id": state["user_id"],
        "session_id": state["session_id"],
        "query_type": answer
    }
def route_query(state: AgentState) -> AgentState:
    """
    Route the query to appropriate tool based on user intent.

    Returns updated state with query_type set.
    """
    messages = state["messages"]
    last_message = messages[-1].content.lower() if messages else ""

    routing_prompt = f"""Bạn là Ami, một trợ lý phân loại ý định để điều phối xử lý.

Hãy phân tích tin nhắn và chọn đúng một trong ba nhãn sau:

1. "excercise_search": Chọn nhãn này trong hai trường hợp:
   - Trường hợp A: Người dùng trực tiếp yêu cầu bài tập, kỹ thuật, hoặc hỏi "làm gì bây giờ".
   - Trường hợp B: Người dùng thể hiện dấu hiệu tâm lý tiêu cực như buồn bã, lo âu, stress, mất ngủ, hoặc bế tắc (Ví dụ: "Mình thấy mệt mỏi quá", "Dạo này mình hay khóc").
2. "knowledge_search": Khi người dùng tìm kiếm định nghĩa hoặc giải thích khoa học thuần túy.
3. "general": Chỉ dành cho chào hỏi xã giao hoặc trò chuyện vui vẻ, tích cực.

Quy tắc ưu tiên: Nếu có bất kỳ dấu hiệu tâm lý tiêu cực nào, hãy LUÔN CHỌN "excercise_search" để Ami có thể đưa ra giải pháp hỗ trợ ngay.

Tin nhắn của người dùng: {last_message}

Chỉ trả lời ĐÚNG MỘT từ: knowledge_search, excercise_search hoặc general."""
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
            model="hoangchihien3011/VietMind", 
            messages=formatted_messages,
            temperature=0.3,
            extra_body={"reasoning_effort": "medium"},
            max_tokens=16000            
    )
    route = response.choices[0].message.content.strip().lower()
            
    # Ensure route is valid
    if route not in ["knowledge_search", "general", "excercise_search"]:
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

def handle_self_harm_emergency(state: AgentState) -> AgentState:
    """Handle self-harm emergency situations."""
    message = state["messages"]
    user_id = state.get("user_id", "default_user")
    tool_result = parallel_knowledge_search.invoke({"query": message[-1].content})
    system_context = f"""{AMI_PERSONALITY}

ĐÂY LÀ TÌNH HUỐNG KHẨN CẤP - Người dùng đang có dấu hiệu tự làm hại bản thân.

YÊU CẦU XỬ LÝ KHẨN CẤP:

1. THÁI ĐỘ:
   - Giữ giọng điệu bình tĩnh, ấm áp và không phán xét
   - Thể hiện sự đồng cảm sâu sắc và quan tâm chân thành
   - Không làm người dùng cảm thấy xấu hổ hay tội lỗi
   - Khẳng định rằng việc họ chia sẻ là rất dũng cảm

2. NỘI DUNG TRẢ LỜI (tuân thủ ĐÚNG THỨ TỰ):
   
   Bước một: Thừa nhận cảm xúc
   - Công nhận cảm xúc và nỗi đau của họ là thật
   - Ví dụ: Mình hiểu bạn đang trải qua khoảng thời gian rất khó khăn và đau đớn
   
   Bước hai: Kết nối và lắng nghe
   - Cho họ biết mình luôn ở đây lắng nghe
   - Nhấn mạnh rằng họ không cô đơn
   
   Bước ba: Can thiệp ngay lập tức
   - Khuyến khích họ liên hệ NGAY với:
     * Đường dây nóng tâm lý: 1800 599 199 (miễn phí 24/7)
     * Người thân, bạn bè đáng tin cậy
     * Cơ sở y tế, bệnh viện gần nhất nếu cần
   - Nhấn mạnh việc tìm giúp đỡ ngay bây giờ là QUAN TRỌNG NHẤT
   
   Bước bốn: Hành động an toàn
   - Khuyên họ tránh xa các vật dụng nguy hiểm
   - Đề xuất ở cùng người thân hoặc bạn bè tin tưởng
   - Gợi ý đến nơi công cộng an toàn nếu ở một mình
   
   Bước năm: Hi vọng và hỗ trợ dài hạn
   - Nhắc nhở rằng cảm giác này sẽ qua đi
   - Khuyến khích tìm chuyên gia tâm lý để được hỗ trợ lâu dài
   - Cam kết mình vẫn ở đây lắng nghe khi họ cần

3. ĐIỀU TUYỆT ĐỐI KHÔNG LÀM:
   - Không nói "mọi chuyện sẽ ổn thôi" một cách hời hợt
   - Không đưa ra lời khuyên chung chung như "hãy suy nghĩ tích cực"
   - Không so sánh với người khác hoặc nói "người khác còn khổ hơn"
   - Không làm nhẹ vấn đề hay nói "bạn đang nghĩ quá"
   - Không chỉ đưa ra bài tập mà không có sự hỗ trợ khẩn cấp

4. THÔNG TIN TỪ KIẾN THỨC:
Kết quả tìm kiếm về hỗ trợ khẩn cấp: {tool_result}

Hãy sử dụng thông tin trên để bổ sung các nguồn lực cụ thể nhưng LUÔN ƯU TIÊN khuyến khích họ tìm sự giúp đỡ chuyên nghiệp NGAY LẬP TỨC.

QUAN TRỌNG: Trả lời trong 6 đến 10 câu, ngắn gọn nhưng đầy đủ các bước trên. Tập trung vào hành động CỤ THỂ và SỐ ĐIỆN THOẠI KHẨN CẤP."""
    
    formatted_messages = [{"role": "system", "content": system_context[:4000]}]  # Limit system context
    current_message = message[-1].content if message else ""
    formatted_messages.append({"role": "user", "content": current_message})
    
    response = openai_client.chat.completions.create(
            model="hoangchihien3011/VietMind", 
            messages=formatted_messages,
            temperature=0.3,
            max_tokens=8000  # Safe limit for output
    )
    return {
        "messages": [AIMessage(content=response.choices[0].message.content)],
        "user_id": user_id,
        "session_id": state["session_id"],
        "query_type": "handle_emergency"
    }

def handle_excercise_search(state: AgentState) -> AgentState:
    messages = state["messages"]
    user_id = state.get("user_id", "default_user")
    dass21_data = query_dass21_scores.invoke({"user_id": user_id, "days": 30})
    query = messages[-1].content

    # Call parallel search tool
    tool_result = parallel_knowledge_search.invoke({"query": query})

    # Generate response using Ami personality with DASS21 context
    system_context = f"""{AMI_PERSONALITY}
Hãy đưa ra các gợi ý bài tập hoặc kỹ thuật thực hành cụ thể dựa trên kết quả tìm kiếm. 
Thay vì viết đoạn văn dài, bạn hãy trình bày dưới dạng danh sách các bước rõ ràng để người dùng dễ theo dõi.
Vì không được dùng ký tự đặc biệt, bạn hãy dùng các từ chỉ thứ tự như Bước một, Bước hai, Bước ba hoặc Một là, Hai là, Ba là để phân tách các ý.
Hãy đi thẳng vào hướng dẫn thực hiện, không dùng ẩn dụ hay nói vòng vo.

Kết quả tìm kiếm: {tool_result}
Dữ liệu DASS21 (tham khảo): {dass21_data}"""
    # Format messages properly with separate roles to avoid token overflow
    formatted_messages = [{"role": "system", "content": system_context[:3000]}]  # Limit system context
    
    # Add current user message
    current_message = messages[-1].content if messages else ""
    formatted_messages.append({"role": "user", "content": current_message})
    
    response = openai_client.chat.completions.create(
            model="hoangchihien3011/VietMind", 
            messages=formatted_messages,
            temperature=0.3,
            max_tokens=8000  # Safe limit for output
    )


    return {
        "messages": [AIMessage(content=response.choices[0].message.content)],
        "user_id": user_id,
        "session_id": state["session_id"],
        "query_type": state["query_type"]
    }
    
    
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

    # Format messages properly with separate roles to avoid token overflow
    formatted_messages = [{"role": "system", "content": system_context[:3000]}]  # Limit system context
    
    # Only include last 3 messages from history
    recent_history = conversation_history[-3:] if len(conversation_history) > 3 else conversation_history
    for msg in recent_history:
        role = "assistant" if isinstance(msg, AIMessage) else "user"
        content = msg.content[:500] if len(msg.content) > 500 else msg.content
        formatted_messages.append({"role": role, "content": content})
    
    # Add current user message
    current_message = messages[-1].content if messages else ""
    formatted_messages.append({"role": "user", "content": current_message})
    
    response = openai_client.chat.completions.create(
            model="hoangchihien3011/VietMind", 
            messages=formatted_messages,
            temperature=0.3,
            max_tokens=8000  # Safe limit for output
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

    # Format messages properly with separate roles to avoid token overflow
    formatted_messages = [{"role": "system", "content": system_context}]
    
    # Only include last 3 messages from history
    recent_history = conversation_history[-3:] if len(conversation_history) > 3 else conversation_history
    for msg in recent_history:
        role = "assistant" if isinstance(msg, AIMessage) else "user"
        content = msg.content[:500] if len(msg.content) > 500 else msg.content
        formatted_messages.append({"role": role, "content": content})
    
    # Add current user message
    current_message = messages[-1].content if messages else ""
    formatted_messages.append({"role": "user", "content": current_message})
    
    response = openai_client.chat.completions.create(
            model="hoangchihien3011/VietMind", 
            messages=formatted_messages,
            temperature=0.3,
            max_tokens=8000  # Safe limit for output
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
    workflow.add_node("self_harm_detector", detech_self_harm_intent)
    workflow.add_node("handle_emergency", handle_self_harm_emergency)
    workflow.add_node("router", route_query)
    workflow.add_node("excercise_handle", handle_excercise_search)
    workflow.add_node("knowledge_handler", handle_knowledge_search)
    workflow.add_node("general_handler", handle_general)

    # Add edges
    workflow.set_entry_point("self_harm_detector")
    workflow.add_conditional_edges(
        "self_harm_detector",
        lambda state: state["query_type"],
        {
            "yes": "handle_emergency",
            "no": "router"
        }   
    )
    # Conditional routing based on query type
    workflow.add_conditional_edges(
        "router",
        lambda state: state["query_type"],
        {
            "knowledge_search": "knowledge_handler",
            "general": "general_handler",
            "excercise_search": "excercise_handle"
        }
    )

    # All handlers end the conversation
    workflow.add_edge("knowledge_handler", END)
    workflow.add_edge("excercise_handle", END)
    workflow.add_edge("general_handler", END)
    workflow.add_edge("handle_emergency", END)

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
