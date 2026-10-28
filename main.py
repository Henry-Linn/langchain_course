from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch

load_dotenv()

class Source(BaseModel):
    """
    Schema for a source used by the agent
    """
    url: str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """
    Schema for an agent response with answer and sources
    """

    answer: str = Field()
    sorces: List[Source] = Field(default_factory=list, description="List of sources used to generate the answer.")

llm = ChatOllama(
        model="qwen2.5:7b",
        temperature=0
    )

tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)

def main():
    print("Hello from langchain-course!")
    result = agent.invoke({"messages":HumanMessage(content="Search 3 AI engineer job postings.")})
    print(result)

if __name__ == "__main__":
    main()
