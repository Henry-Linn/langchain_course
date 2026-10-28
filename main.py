from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from tavily import TavilyClient

load_dotenv()
tavily = TavilyClient()

@tool
def search(query: str) -> str:
    """Search the web for current information, including weather, news, and facts.
   Always use this for questions about current conditions."""
    print(f"Searching for {query}")
    return tavily.search(query=query)

llm = ChatOllama(
        model="qwen2.5:7b",
        temperature=0
    )
tools = [search]
agent = create_agent(model=llm, tools=tools)

def main():
    print("Hello from langchain-course!")
    result = agent.invoke({"messages":HumanMessage(content="Search 3 AI engineer job postings.")})
    print(result)



if __name__ == "__main__":
    main()
