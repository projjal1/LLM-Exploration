from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub
from langchain.agents import create_tool_calling_agent, AgentExecutor
import os

# Ignore deprecation warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load config from .env
load_dotenv()

# Create LLM instance
llm = init_chat_model("mistral:7b", model_provider="ollama")

# Graph state
class State(TypedDict):
    topic: str
    essay_draft: str
    essay_improvement: str
    improved_essay: str
    final_essay: str

# Langchain structured response
class EssayResponse(BaseModel):
    essay: str = Field(..., description="A short essay about the given topic")

# Agent to fetch updated information about topic from the internet
search = TavilySearchResults(max_results = 2, tavily_api_key=os.environ.get("TAVILY_API_KEY"))
# Add search plugin to the tool
tools = [search]

# Structure essay response
structured_essay_llm = llm.with_structured_output(EssayResponse)

# define agent to use the tool
prompt = hub.pull("hwchase17/openai-functions-agent")
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

# Nodes
def generate_essay_draft(state: State) -> dict:
    """First LLM call to generate initial essay draft"""

    agent_response = agent_executor.invoke({"input": f"Write a short essay draft of 200 words about {state['topic']} without title."})  
    essay_draft = agent_response['output'].replace("\n", " ")
    return {"essay_draft": essay_draft}

# Check if essay needs improvement for news publication
def check_essay(state: State) -> str:
    """Gate function to check if the essay needs improvement for news publication"""

    # Simple check - does the essay contain "newsworthy" or "important"
    msg = llm.invoke(f"Does this essay need improvement for news publication, answer only yes/no?: {state['essay_draft']}")
    if "YES" in msg.content.upper():
        return "Yes"
    else:
        return "No"
    
# Improve the essay 
def improve_essay(state: State) -> dict:
    """Second LLM call to improve the essay for news publication"""

    msg = structured_essay_llm.invoke(f"Improve this essay for news publication: {state['essay_draft']}")
    improved_essay = msg.essay.replace("\n", " ")
    return {"improved_essay": improved_essay}

def polish_essay(state: State) -> dict:
    """Third LLM call for final polish"""
    msg = structured_essay_llm.invoke(f"Add a twist to the essay at the end: {state['improved_essay']}")
    essay = msg.essay.replace("\n", " ")
    return {"final_essay": essay}


# Build workflow
workflow = StateGraph(State)

# Add nodes
workflow.add_node("generate_essay_draft", generate_essay_draft)
workflow.add_node("improve_essay", improve_essay)
workflow.add_node("polish_essay", polish_essay)

# Add edges to connect nodes
workflow.add_edge(START, "generate_essay_draft")
workflow.add_conditional_edges(
    "generate_essay_draft", check_essay, {"Yes": "improve_essay", "No": END}
)
workflow.add_edge("improve_essay", "polish_essay")
workflow.add_edge("polish_essay", END)

# Compile
chain = workflow.compile()

# Invoke
state = chain.invoke({"topic": "state of non-fungible tokens in 2025"})
print("Initial essay draft:")
print(state["essay_draft"])
print("\n--- --- ---\n")
if "improved_essay" in state:
    print("Improved essay:")
    print(state["improved_essay"])
    print("\n--- --- ---\n")

    print("Final essay:")
    print(state["final_essay"])
else:
    print("Essay failed quality gate - no improvement needed!")