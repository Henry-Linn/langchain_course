import os, requests
from dotenv import load_dotenv
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaEmbeddings
from pinecone import Pinecone, ServerlessSpec

load_dotenv()  # .env wins over stale shell/VS Code variables

if __name__ == '__main__':

    index_name = os.environ["INDEX_NAME"]
    print("Using index:", index_name)

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

    # Create a classic dense index if it doesn't exist yet
    existing = [i["name"] for i in pc.list_indexes()]
    if index_name not in existing:
        pc.create_index(
            name=index_name,
            dimension=768,  # embeddinggemma
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print("Created index", index_name)

    # Inspect it: should show dimension 768 and no document schema
    print(pc.describe_index(index_name))

    # ------ Data Indexing ------
    print("Ingesting...")
    loader = UnstructuredLoader(
        file_path="mediumblog1.txt",
        encoding="UTF-8",
    )
    # It is going to split the document into metadata per line.
    document = loader.load()

    print("splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")
    embeddings = OllamaEmbeddings(model="embeddinggemma")

    print("ingesting...")
    PineconeVectorStore.from_documents(texts, embeddings, index_name=index_name)
    print("finish")