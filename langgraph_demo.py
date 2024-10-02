import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END , StateGraph, MessagesState
from langchain_core.messages import HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from typing import Literal
from dotenv import load_dotenv
load_dotenv()

os.environ["LANGCHAIN_ENDPOINT"] = os.getenv('LANGCHAIN_ENDPOINT')
os.environ["LANGCHAIN_API_KEY"] = os.getenv('LANGCHAIN_API_KEY')
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv('LANGCHAIN_TRACING_V2')
os.environ["LANGCHAIN_PROJECT"] = os.getenv('LANGCHAIN_PROJECT')
os.environ['TAVILY_API_KEY'] = os.getenv('TAVILY_API_KEY')
google_key = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-pro",api_key=google_key, 
                             temperature=0.7)


# @tool
# def search(query:str):
#     """Call the stuff the web"""
#     if "sf" in query.lower() or "san francisco" in query.lower():
#         return "It's 60 degrees and foggy."
#     return "It's 90 degrees and sunny."

tool = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
    include_images=True,
    # include_domains=[...],
    # exclude_domains=[...],
    # name="...",            # overwrite default tool name
    # description="...",     # overwrite default tool description
    # args_schema=...,       # overwrite default args_schema: BaseModel
)

tools=ToolNode([tool])

def call_model(state:MessagesState):
    messages = state["messages"]
    response = llm.invoke(messages)
    return {'messages':response}

def should_continue(state: MessagesState) -> Literal['tools', END]:
    messages = state["messages"]
    last_message = messages[-1]

    if last_message.tool_calls:
        return "tools"

    return END

work_flow = StateGraph(MessagesState)
work_flow.add_node("agent",call_model)
work_flow.add_node("tools", tools)
work_flow.add_conditional_edges("agent",should_continue)
work_flow.set_entry_point("agent")
work_flow.add_edge("agent","tools") 

memory =  MemorySaver()
app = work_flow.compile(checkpointer=memory)

result=app.invoke(input={"messages":HumanMessage(content="what is the weather in Ha Noi")},
           config={"configurable":{"thread_id":42}})

print(result['messages'][-1].content)