from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from rich.console import Console
from tavily import TavilyClient

load_dotenv()

tavily = TavilyClient()

@tool
def search(query: str) -> dict:
    """
    Tool that searches the internet
    Args:
        query (str): search query
    Returns:
        str: search result
    """
    print(f"Searching internet for {query}")
    return tavily.search(query=query)

llm = ChatOpenAI(model="gpt-4o-mini")
tools = [search]

agent = create_agent(model=llm, tools=tools)

# noinspection PyTypeChecker
result = agent.invoke({"messages": HumanMessage(content="List 3 jobs posting in Hyderabad for LLM Engineering")})

console = Console()
if "messages" in result:
    console.print(result["messages"][-1].content)
else:
    console.print(result)