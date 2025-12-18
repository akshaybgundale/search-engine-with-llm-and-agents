import streamlit as st
import operator
from typing import TypedDict, Annotated, List

from langchain_groq import ChatGroq
from langchain_community.tools import (
    DuckDuckGoSearchRun,
    WikipediaQueryRun,
    ArxivQueryRun
)
from langchain_community.utilities import (
    WikipediaAPIWrapper,
    ArxivAPIWrapper
)

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    BaseMessage
)

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# -------------------------------------------------
# Tools
# -------------------------------------------------
search_tool = DuckDuckGoSearchRun()

wiki_tool = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(
        top_k_results=1,
        doc_content_chars_max=400
    )
)

arxiv_tool = ArxivQueryRun(
    api_wrapper=ArxivAPIWrapper(
        top_k_results=1,
        doc_content_chars_max=400
    )
)

tools = [search_tool, wiki_tool, arxiv_tool]

# -------------------------------------------------
# Graph State (FIXED reducer)
# -------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]

# -------------------------------------------------
# Streamlit UI
# -------------------------------------------------
st.title("🔎 LangGraph – Chat with Web Search")

api_key = st.sidebar.text_input("Enter your Groq API Key", type="password")

if "messages" not in st.session_state:
    st.session_state.messages = [
        AIMessage(content="Hello! I can search the web for you.")
    ]

for msg in st.session_state.messages:
    role = "assistant" if isinstance(msg, AIMessage) else "user"
    st.chat_message(role).write(msg.content)

user_input = st.chat_input("Ask something")

# -------------------------------------------------
# LangGraph Nodes
# -------------------------------------------------
def llm_node(state: AgentState) -> AgentState:
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.1-8b-instant"
    )

    llm_with_tools = llm.bind_tools(tools)

    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

tool_node = ToolNode(tools)

def route(state: AgentState):
    last_msg = state["messages"][-1]
    if getattr(last_msg, "tool_calls", None):
        return "tools"
    return END

# -------------------------------------------------
# Build Graph
# -------------------------------------------------
graph = StateGraph(AgentState)

graph.add_node("llm", llm_node)
graph.add_node("tools", tool_node)

graph.set_entry_point("llm")
graph.add_conditional_edges("llm", route)
graph.add_edge("tools", "llm")

app = graph.compile()

# -------------------------------------------------
# Execute
# -------------------------------------------------
if user_input and api_key:
    human_msg = HumanMessage(content=user_input)
    st.session_state.messages.append(human_msg)
    st.chat_message("user").write(user_input)

    result = app.invoke({"messages": st.session_state.messages})

    st.session_state.messages = result["messages"]
    final_msg = st.session_state.messages[-1]

    st.chat_message("assistant").write(final_msg.content)
