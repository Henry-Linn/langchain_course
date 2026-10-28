from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

load_dotenv()

@tool
def search(query: str) -> str:
    """Search the customer database for records matching the query.

    Args:
        query: Search terms to look for
        limit: Maximum number of results to return
    """
    print(f"Searching for {query}")
    return "Tokyo weather is sunny."

llm = ChatOllama()
tools = [search]
agent = create_agent(model=llm, tools=tools)

def main():
    print("Hello from langchain-course!")
    result = agent.invoke({"message":HumanMessage(content="What is the weather in Tokyo?")})
    print(result)


if __name__ == "__main__":
    main()
