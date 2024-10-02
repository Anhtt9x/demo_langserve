from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from fastapi import FastAPI
import uvicorn
from dotenv import load_dotenv
import os
from langserve import add_routes
load_dotenv()

google_key = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-8b-exp-0924",api_key=google_key)

prompt = ChatPromptTemplate.from_messages([("system","you are a helpfull chatbot"),("human","{question}")])

app = FastAPI(title="My API", description="demo app langserve", version="0.1.0")

chain = prompt |  llm | StrOutputParser()

add_routes(app,chain,path="/chain")

if  __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)