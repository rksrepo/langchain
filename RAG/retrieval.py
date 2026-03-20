import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_pinecone import PineconeVectorStore
from rich.console import Console

load_dotenv(override=True)

print("Initializing components...")

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI()

vector_store = PineconeVectorStore(
    embedding=embeddings,
    index_name=os.environ["INDEX_NAME"]
)

retriever: VectorStoreRetriever = vector_store.as_retriever(search_kwargs={"k": 3})

prompt_template = ChatPromptTemplate.from_template(
    """
    Answer the question using only the information provided in the context below.

    If the answer is not explicitly present in the context, say "I don't know based on the provided context."

    Do not make up information or use outside knowledge.
    
    You must answer strictly using only the provided context.

        - Do NOT use any prior knowledge.
        - Do NOT infer beyond what is explicitly stated.
        - If the answer is not clearly available, respond with: "I don't know based on the provided context."
        - Do NOT guess or hallucinate.

    {context}

    Question: {question}

    Provide a detailed answer:
    """
)

def format_docs(docs):
    """Format retrieved documents into a single string"""
    return "\n\n".join(doc.page_content for doc in docs)

def retrieval_without_lcel(query: str):
    docs = retriever.invoke(query)

    context = format_docs(docs)

    messages = prompt_template.format_messages(context=context, question=query)

    response = llm.invoke(messages)

    return response.content


if __name__ == "__main__":
    print("Retrieving ...")

    query = "What is Karthik's role in Mahabharata"

    result_without_lcel = retrieval_without_lcel(query)

    console = Console()
    console.print(result_without_lcel)
