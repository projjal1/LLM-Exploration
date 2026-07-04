from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model

# Create LLM instance
llm = init_chat_model("mistral:7b", model_provider="ollama")

# Graph state
class State(TypedDict):
    topic: str
    joke: str
    possible_improvement: str
    improved_joke: str
    final_joke: str

# Langchain structured response
class JokeResponse(BaseModel):
    joke: str = Field(..., description="A short joke about the given topic")

# Structure joke response
structured_joke_llm = llm.with_structured_output(JokeResponse)

# Nodes
def generate_joke(state: State) -> dict:
    """First LLM call to generate initial joke"""

    msg = structured_joke_llm.invoke(f"Write a short joke about {state['topic']}.")
    joke = msg.joke.replace("\n", " ")
    return {"joke": joke}


def check_punchline(state: State) -> str:
    """Gate function to check if the joke has a punchline"""

    # Simple check - does the joke contain "?" or "!"
    msg = llm.invoke(f"Can this joke be improved, answer only yes/no?: {state['joke']}")
    if "YES" in msg.content.upper():
        return "Yes"
    else:
        return "No"
    

def improve_joke(state: State) -> dict:
    """Second LLM call to improve the joke"""

    msg = structured_joke_llm.invoke(f"Make this joke funnier by adding wordplay: {state['joke']}")
    joke = msg.joke.replace("\n", " ")
    return {"improved_joke": joke}


def polish_joke(state: State) -> dict:
    """Third LLM call for final polish"""
    msg = structured_joke_llm.invoke(f"Add a surprising twist to the existing joke at the end: {state['improved_joke']}")
    joke = msg.joke.replace("\n", " ")
    return {"final_joke": joke}


# Build workflow
workflow = StateGraph(State)

# Add nodes
workflow.add_node("generate_joke", generate_joke)
workflow.add_node("improve_joke", improve_joke)
workflow.add_node("polish_joke", polish_joke)

# Add edges to connect nodes
workflow.add_edge(START, "generate_joke")
workflow.add_conditional_edges(
    "generate_joke", check_punchline, {"Yes": "improve_joke", "No": END}
)
workflow.add_edge("improve_joke", "polish_joke")
workflow.add_edge("polish_joke", END)

# Compile
chain = workflow.compile()

# Invoke
state = chain.invoke({"topic": "cats"})
print("Initial joke:")
print(state["joke"])
print("\n--- --- ---\n")
if "improved_joke" in state:
    print("Improved joke:")
    print(state["improved_joke"])
    print("\n--- --- ---\n")

    print("Final joke:")
    print(state["final_joke"])
else:
    print("Joke failed quality gate - no punchline detected!")