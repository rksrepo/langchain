from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from rich.console import Console

@tool
def search(query: str) -> str:
    """
    Tool that searches the internet
    Args:
        query (str): search query
    Returns:
        str: search result
    """
    print(f"Searching internet for {query}")
    return "Hyderabad weather is very sunny"

llm = ChatOpenAI(model="gpt-4o-mini")
tools = [search]

agent = create_agent(model=llm, tools=tools)

# noinspection PyTypeChecker
result = agent.invoke({"messages": HumanMessage(content="How is the weather in Hyderabad")})

console = Console()
print(console.print(result))