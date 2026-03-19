import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

load_dotenv(override=True)

MAX_ITERATIONS = 10
MODEL = "gpt-4o-mini"

# Tools required


@tool
def get_product_price(product: str) -> float:
    """
    Retrieve the price of a product from a predefined catalog.

    Notes:
        - Use this tool when a user asks for the price of a specific product.
        - Input should match one of the supported product names.

    """
    print(f" >> Executing get_product_price for {product}")

    product_prices = {
        "laptop": 75000.00,
        "mouse": 499.99,
        "keyboard": 1499.50,
        "usb_stick": 799.00,
        "headset": 1999.99,
    }

    return product_prices[product]


@tool
def calculate_price_with_tier(price: float, tier: str) -> float:
    """
    Calculate final price based on discount tier.

    Tiers:
    - "bronze": 10% discount
    - "silver": 20% discount
    - "gold": 25% discount

    """

    print(f" >> Executing calculate_price_with_tier for {tier} and price: {price}")

    tier_map = {
        "bronze": 10,
        "silver": 20,
        "gold": 25
    }

    if price < 0:
        raise ValueError("Price cannot be negative")

    if tier.lower() not in tier_map:
        raise ValueError("Invalid tier. Choose bronze, silver, or gold")

    discount = tier_map[tier.lower()]
    final_price = price * (1 - discount / 100)

    return final_price


# noinspection PyTypeChecker
@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    tools = [get_product_price, calculate_price_with_tier]
    tools_dict = {t.name: t for t in tools}

    llm = init_chat_model(f"openai:{MODEL}")
    llm_with_tools = llm.bind_tools(tools)

    print(f"Question: {question}")

    messages = [
        SystemMessage(
            "You are a helpful shopping assistant. You have access to a product catalog through tool calling."
            "You also have access to a discount tool. \n\n"
            "STRICT RULES - YOU MUST FOLLOW THIS VERY STRICTLY:"
            "1. Never guess or assume any product price"
            "You must first call get_product_price to get real price of product"
            "2. Only call calculate_price_with_tier after you have a received a price."
            "Customer can request only for 3 tiers, if giving a tier that does not exist, then return only the actual price."
            "3. Never calculate any discounts yourself, use only the tools given to do so."
            "4. If the user does not provide any tier, ask for it, do no assume any tier by yourself"
        ),

        HumanMessage(
            content=question,
        )
    ]

    for iterations in range(1, MAX_ITERATIONS + 1):
        print(f"\n -- Iteration {iterations} --")
        ai_message = llm_with_tools.invoke(messages)
        tools_calls = ai_message.tool_calls

        if not tools_calls:
            print(f"\n Final Answer: {ai_message.content}")
            return ai_message.content

        tool_call = tools_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f" Tool Selected {tool_name} with args {tool_args}. Id: {tool_call_id}")
        tool_to_use = tools_dict.get(tool_name)

        if tool_to_use is None:
            raise ValueError(f"Tool {tool_name} not found in tools dict")

        observation = tool_to_use.invoke(tool_args)

        print(f"Tool Result: {observation}")

        messages.append(ai_message)
        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )

    return None


if __name__ == "__main__":
    result = run_agent("What is the price for a laptop - give me original price and discounted price with silver tier.?")
    print(result)