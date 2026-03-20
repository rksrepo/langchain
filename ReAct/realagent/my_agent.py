from dotenv import load_dotenv
from langchain.agents import create_agent
from my_tools import (apply_birthday_discount, calculate_discount,
                      get_product_price)
from prompts import system_prompts, user_prompt1, user_prompt2
from rich.console import Console

load_dotenv()
console = Console()

agent = create_agent(
    model="gpt-4o-mini",
    tools=[get_product_price, calculate_discount, apply_birthday_discount],
)

# Run
# noinspection PyTypeChecker
response1 = agent.invoke({"messages": [system_prompts, user_prompt1]})
final1 = response1["messages"][-1].content

# noinspection PyTypeChecker
response2 = agent.invoke({"messages": [system_prompts, user_prompt2]})
final2 = response2["messages"][-1].content

console.print(final1)
print("***************************")
console.print(final2)
