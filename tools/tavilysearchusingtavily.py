from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from rich.console import Console

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")
tools = [TavilySearch()]

agent = create_agent(model=llm, tools=tools)

# noinspection PyTypeChecker
result = agent.invoke(
    {
        "messages": HumanMessage(
            content="List 3 jobs posting in Hyderabad for LLM Engineering"
        )
    }
)

console = Console()
if "messages" in result:
    console.print(result["messages"][-1].content)
else:
    console.print(result)
