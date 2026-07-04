from langchain.agents import create_agent 
from tools import lookup_order, search_faq, create_support_ticket

# Create an agent with the tools and a system prompt
agent = create_agent(
    model="ollama:qwen3:8b",
    tools=[lookup_order, search_faq, create_support_ticket],
    system_prompt="""
You are a customer support assistant.

You have access to tools.

IMPORTANT:

Never explain that you are going to use a tool.

Never say things like:

"I'll use lookup_order."

"I will call the tool."

"I'll check the database."

Instead immediately call the tool.

Rules:

If an order number is mentioned,
always call lookup_order.

If the user asks about passwords,
refunds or shipping,
always call search_faq.

If the user reports a damaged,
broken or defective product,
always call create_support_ticket.

Only answer after the tool returns.

Be friendly.
"""
)

# Sample questions to test the agent
questions = ["Where is order 1002?", "How do I reset my password?", "I received a damaged item. The screen of the laptop is cracked. Kindly create a support ticket for me."]

# Invoke the agent with each question and print the response
for q in questions:
    print(f"Question: {q}")
    response = agent.invoke({
        "messages": [{"role": "user", "content": q}]
    })
    print(f"Response: {response['messages'][-1].content}")
    print("=" * 50)