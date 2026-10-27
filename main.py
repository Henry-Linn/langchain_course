from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
# load all the enviroment variables; All the keys for calling APIs
load_dotenv()

def main():
    print("Hello from langchain-course!")

    information  = """
    A black cat is a domestic cat with black fur. They may be a specific breed, or a common domestic cat of no particular breed. Most black cats have golden irises due to their high melanin pigment content. Black cats are the subject of mythology, legend, and superstition. They are often associated with witches and good or bad luck in European folklore. 
    The Cat Fanciers' Association (CFA) recognizes 22 cat breeds that can come with solid black coats.[1] The Bombay breed is exclusively black.
    """

    summary_template = """
    given the information {information} about an animal I want you to create:
    1. A short summary
    2. two interesting facts about them 
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    # Initialize the local model
    llm = ChatOllama(
        model="qwen2.5:14b",
        temperature=0
    )

    # Pass the prompt to the LLM. This line will create a runnable object.
    chain = summary_prompt_template | llm

    # Send a prompt to the model
    response = chain.invoke(input={"information": information})

    # Print the response
    print(response.content)
    

if __name__ == "__main__":
    main()
