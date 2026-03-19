from dotenv import load_dotenv
from langchain.agents.structured_output import ProviderStrategy
from langchain.agents import create_agent
from rich.console import Console

load_dotenv(override=True)

contact_info_schema = {
    "title": "extract_contact_info",
    "type": "object",
    "description": "Contact information for a person.",
    "properties": {
        "name": {"type": "string", "description": "The name of the person"},
        "email": {"type": "string", "description": "The email address of the person"},
        "phone": {"type": "string", "description": "The phone number of the person"}
    },
    "required": ["name", "email", "phone"]
}

agent = create_agent(
    model="gpt-4o-mini",
    response_format=ProviderStrategy(contact_info_schema)
)

# noinspection PyTypeChecker
result = agent.invoke({
    "messages": [{"role": "user", "content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567"}]
})

console = Console()
console.print(result["structured_response"])


