import os
from typing import List

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field
from rich.console import Console

load_dotenv(override=True)


class Source(BaseModel):
    """
    Schema for a Source used by Agent
    """

    url: str = Field(..., description="The URL for the source")


class AgentResponse(BaseModel):
    """
    Schema for a Response used by Agent with answers and sources
    """

    answer: str = Field(..., description="The answer to the user query")
    source: List[Source] = Field(
        default_factory=list, description="The sources that the user answered"
    )


llm = ChatOpenAI(model="gpt-4o-mini")
tools = [TavilySearch(model="gpt4o")]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)

# noinspection PyTypeChecker
result = agent.invoke(
    {"messages": HumanMessage("3 jobs posting in Hyderabad for Java senior engineer")}
)

console = Console()

answer = result["structured_response"].answer
source = result["structured_response"].source

console.print(source)
console.print(answer)
