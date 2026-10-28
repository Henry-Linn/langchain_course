from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

load_dotenv()

def main():
    print("Hello from langchain-course!")

if __name__ == "__main__":
    main()
