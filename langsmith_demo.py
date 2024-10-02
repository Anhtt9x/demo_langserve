from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
import os
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt.chat_agent_executor import create_react_agent
from dotenv import load_dotenv
load_dotenv()

os.environ["LANGCHAIN_ENDPOINT"] = os.getenv('LANGCHAIN_ENDPOINT')
os.environ["LANGCHAIN_API_KEY"] = os.getenv('LANGCHAIN_API_KEY')
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv('LANGCHAIN_TRACING_V2')
os.environ["LANGCHAIN_PROJECT"] = os.getenv('LANGCHAIN_PROJECT')

google_key = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-8b-exp-0924",api_key=google_key)

prompt = ChatPromptTemplate.from_messages([("system","you are a helpfull chatbot"),("human","{question}")])

chain = prompt | llm | StrOutputParser()

memory = MemorySaver()

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

agent_executor = create_react_agent(model=llm, tools=[tool], checkpointer=memory)

config = {"configurable":{"thread_id":"abc123"}}

for chunk in agent_executor.stream(input={"messages":HumanMessage(content="hi iam sunny! and i live in Vietnam, how is weather in my country")},config=config):
    print(chunk)
    print("----------")


