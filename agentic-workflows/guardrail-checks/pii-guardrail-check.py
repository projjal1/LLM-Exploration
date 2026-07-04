'''
Sample script to validate PII masking and redaction for user prompt when passed to tool.
'''


from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware
from langchain.tools import tool


# Tool to log the output 
@tool
def log_content(content: str) -> str:
    """Logs the content passed to the tool."""
    
    print("=" * 50)
    print("TOOL RECEIVED:")
    print(content)
    print("=" * 50)

    return f"Logged content: {content}"

agent = create_agent(
    model="ollama:mistral:7b",
    tools=[log_content],
    system_prompt="Log the user prompt for tracking usage.",
    middleware=[
        # Redact emails in user input before sending to model
        PIIMiddleware(
            "email",
            strategy="redact",
            apply_to_input=True,
        ),
        # Mask credit cards in user input
        PIIMiddleware(
            "credit_card",
            strategy="mask",
            apply_to_input=True,
        )
    ],
)

# When user provides PII, it will be handled according to the strategy
result = agent.invoke({
    "messages": [{"role": "user", "content": "My email is john.doe@example.com and card is 5105-1051-0510-5100"}]
})
print(result["messages"][-1].content)


'''
OUTPUT 

==================================================
TOOL RECEIVED:                                                  
User prompt: My email is REDACTED_EMAIL and card is ****-****-****-5100
==================================================
 I'm unable to assist you with personal information such as your email or credit card number. However, it's great that you're aware of the importance of keeping sensitive data secure! If you have any other non-personal questions or need assistance with something else, feel free to ask.
'''