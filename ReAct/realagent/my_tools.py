from langchain.tools import tool


@tool
def get_product_price(product: str) -> float:
    """Returns price of a product"""
    prices = {"laptop": 50000, "mobile": 20000}
    return prices.get(product.lower(), 0)


@tool
def calculate_discount(price: float, tier: str) -> float:
    """Applies discount based on tier (silver/gold)"""
    discounts = {"silver": 0.1, "gold": 0.2}
    discount = discounts.get(tier.lower(), 0)
    return price * (1 - discount)


@tool
def apply_birthday_discount(price: float, is_birthday: bool) -> float:
    """Apply extra 5% discount if it's user's birthday"""
    if is_birthday:
        return price * 0.95
    return price
