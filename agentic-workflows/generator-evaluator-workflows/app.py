from pydantic import BaseModel, Field
from typing_extensions import Literal, TypedDict
from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model

# Create LLM instance
llm = init_chat_model("mistral:7b", model_provider="ollama")

# Graph state
class State(TypedDict):
    email: str
    email_request: str
    feedback: str
    approval_status: str


# Schema for structured output to use in evaluation
class Feedback(BaseModel):
    grade: Literal["professional", "not professional"] = Field(
        description="Decide if the drafted email is professional and ready to send.",
    )
    feedback: str = Field(
        description="If the email is not ready, provide specific feedback on how to improve clarity, tone, structure, or professionalism.",
    )


# Augment the LLM with schema for structured output
evaluator = llm.with_structured_output(Feedback)


# Nodes
def llm_call_generator(state: State):
    """LLM drafts or revises a professional email based on the user's request."""

    print("*-- Generating Answer ---*\n")
    if state.get("feedback"):
        msg = llm.invoke(
            f"Draft a professional email based on the request: {state['email_request']}. Revise it by applying this feedback: {state['feedback']}."
        )
        print(f"Revised email generated after feedback: {msg.content}")
    else:
        msg = llm.invoke(f"Draft a simple, brief incomplete email with some intentional errors for the request: {state['email_request']}. Keep it basic since this is an initial attempt.")
        print(f"Initial email draft generated: {msg.content}")
    return {"email": msg.content}

def llm_call_evaluator(state: State):
    """LLM evaluates the drafted email and provides improvement feedback if required."""

    grade = evaluator.invoke(f"Evaluate whether this drafted email is professional, adheres to a strictly formal tone and ready to send based on the request: {state['email_request']}. Email: {state['email']}")
    print("*-- Evaluation Result ---*\n")
    print(f"Email evaluation grade: {grade.grade}, feedback: {grade.feedback}")
    return {"approval_status": grade.grade, "feedback": grade.feedback}

# Conditional edge function to route back to answer generator or end based upon feedback from the interviewer
def route_answer(state: State):
    """Route based on whether the drafted email is approved or needs revision."""

    if state["approval_status"] == "professional":
        return "Accepted"
    elif state["approval_status"] == "not professional":
        return "Rejected + Feedback"


# Build workflow
optimizer_builder = StateGraph(State)

# Add the nodes
optimizer_builder.add_node("llm_call_generator", llm_call_generator)
optimizer_builder.add_node("llm_call_evaluator", llm_call_evaluator)

# Add edges to connect nodes
optimizer_builder.add_edge(START, "llm_call_generator")
optimizer_builder.add_edge("llm_call_generator", "llm_call_evaluator")
optimizer_builder.add_conditional_edges(
    "llm_call_evaluator",
    route_answer,
    {  # Name returned by route_answer : Name of next node to visit
        "Accepted": END,
        "Rejected + Feedback": "llm_call_generator",
    },
)

# Compile the workflow
optimizer_workflow = optimizer_builder.compile()

# Show the workflow
# display(Image(optimizer_workflow.get_graph().draw_mermaid_png()))

# Invoke
state = optimizer_workflow.invoke({"email_request": "Send a follow‑up email to a client requesting an update on the project status."})
print("\n*-- Final Drafted Email --*\n")
print(state["email"])