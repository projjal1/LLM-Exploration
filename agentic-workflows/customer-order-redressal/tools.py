from langchain.tools import tool

# Fake database of orders
ORDERS = {
    "1001": {"status": "Processing", "eta": "2 days"},
    "1002": {"status": "Shipped", "eta": "Tomorrow"},
    "1003": {"status": "Delivered", "eta": "Delivered yesterday"},
}

# Fake collection of common FAQs
FAQ = {
    "password": "To reset your password, click 'Forgot Password' on the login page.",
    "refund": "Refunds are processed within 5 business days.",
    "shipping": "Standard shipping takes 3-5 business days.",
}


# Tool to look up an order by its order ID
@tool
def lookup_order(order_id: str) -> str:
    """
    Retrieves the status of an existing order.

    Use this tool whenever the user asks:
    - Where is my order?
    - Track my package
    - Order status
    - Delivery status

    Input:
        order_id: customer's order number

    Returns:
        Current order status and ETA.
    """

    if order_id not in ORDERS:
        return "Order not found."

    order = ORDERS[order_id]

    return f"""
    Order ID: {order_id}
    Status: {order['status']}
    Estimated delivery: {order['eta']}
    """

# Tool to search the FAQ for a given question
@tool
def search_faq(question: str) -> str:
    """
    Search the company FAQ.

    MUST use this tool whenever the user asks about:

    - password
    - refund
    - shipping
    - account

    Do not answer these questions yourself.
    """

    question = question.lower()

    for key, value in FAQ.items():
        if key in question:
            return value

    return "No FAQ entry found."


# Limit the number of support tickets created to avoid spamming
ticket_counter = 100

# Tool to create a support ticket for a customer
@tool
def create_support_ticket(issue: str) -> str:
    """
    Create a customer support ticket.

    ALWAYS use this tool whenever the customer says:

    - damaged
    - broken
    - defective
    - cracked
    - not working
    - wants a replacement

    Do not ask for additional information.
    Create the ticket immediately.
    """

    global ticket_counter

    ticket_counter += 1

    return f"Support ticket #{ticket_counter} created.\nIssue: {issue}."
