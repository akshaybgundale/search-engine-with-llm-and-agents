import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv

###### Arxiv and Wikipedia tools###########
# Inbuilt tool of arxiv
api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=400)
arxiv_tool = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

# Inbuilt tool of wikipedia
api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=400)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

search=DuckDuckGoSearchRun(name="Search", description="Useful for when you need to look up current events or search the web for specific information.")
###### Arxiv and Wikipedia tools###########

st.title("🔎 Langchain - Chat with WebSearch")

## Side bar for setting
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key", type="password")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hello! I am your assistant who can search on the web. How may I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

prompt=st.chat_input(placeholder= "What is machine learning?")
if prompt and api_key:
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_mesage("user").write(prompt)

    llm= ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True)

    tools = [arxiv_tool, wiki_tool, search]

    search_agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handling_parsing_errors=True,
        verbose=True,
    )

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        # pass the user prompt (string) to the agent
        response = search_agent.run(prompt, callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)