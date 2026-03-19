from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from rich.console import Console

load_dotenv(override=True)

class ContactInfo(BaseModel):
    """Contact information for a person."""
    name: str = Field(description="The name of the person")
    email: str = Field(description="The email address of the person")
    phone: str = Field(description="The phone number of the person")

agent = create_agent(
    model="gpt-4o-mini",
    response_format=ContactInfo  # Auto-selects ProviderStrategy
)

# noinspection PyTypeChecker
result = agent.invoke({
    "messages": [{"role": "user", "content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567"}]
})

console = Console()
console.print(result["structured_response"])


