import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter

load_dotenv(override=True)

if __name__ == "__main__":
    print(">>>> STARTED -- Ingesting Document >>>>")

    loader = TextLoader(
        file_path="mahabharata_rag_large.txt",
        encoding="utf-8",
        autodetect_encoding=True,
    )
    document = loader.load()

    print(">>>> ENDED -- Ingesting Document >>>>\n")

    print(">>>> STARTED -- Splitting Documents >>>>")

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f">>>> Created {len(texts)} chunks >>>>")

    print(">>>> ENDED -- Splitting Documents >>>>\n")

    print(">>>> STARTED -- Embedding >>>>")

    print(f"Ingesting in Pinecone")
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    PineconeVectorStore.from_documents(
        texts, embeddings, index_name=os.environ["INDEX_NAME"]
    )
    print(f"Ingesting completed in Pinecone")

    print(">>>> ENDED -- Embedding >>>>\n")
