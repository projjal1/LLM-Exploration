# Using LangGraph to build a state machine for SQL query generation and execution

from typing_extensions import runtime

from accelerate import state
from langchain_community.utilities import SQLDatabase
from states import State, QueryOutput
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool

# Initialize the SQL database connection
db = SQLDatabase.from_uri("sqlite:///Chinook.db")

# Initialize the language model
llm = init_chat_model("llama3:8b", model_provider="ollama")

# Define the prompt template for generating SQL queries
system_message = """
Given an input question, create a syntactically correct {dialect} query to
run to help find the answer. Unless the user specifies in his question a
specific number of examples they wish to obtain, always limit your query to
at most {top_k} results. You can order the results by a relevant column to
return the most interesting examples in the database.

Never query for all the columns from a specific table, only ask for a the
few relevant columns given the question.

Pay attention to use only the column names that you can see in the schema
description. Be careful to not query for columns that do not exist. Also,
pay attention to which column is in which table.

Only use the following tables:
{table_info}
"""

# Define the prompt template for user questions
user_prompt = "Question: {input}"

# Create the complete chat prompt template
query_prompt_template = ChatPromptTemplate(
    [("system", system_message), ("user", user_prompt)]
)

# Write function to generate SQL query
def write_query(state: State):
    """Generate SQL query to fetch information."""  
    prompt = query_prompt_template.invoke(
        {
            "dialect": db.dialect,
            "top_k": 10,
            "table_info": db.get_table_info(),
            "input": state["question"],
        }
    )
    structured_llm = llm.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt)
    return {"query": result["query"]}

# Write function to execute SQL query
def execute_query(state: State):
    """Execute SQL query and return the result. If error occurs, return an error message."""
    execute_query_tool = QuerySQLDatabaseTool(db=db)

    try: 
        return {"result": execute_query_tool.invoke(state["query"])}
    except Exception as e:
        return {"result": "Error executing query"}
    

# Convert SQL result to natural language answer
def generate_answer(state: State):
    """Answer question using retrieved information as context."""
    prompt = (
        "Given the following user question, corresponding SQL query, "
        "and SQL result, answer the user question. "
        "Rephrase the answer to make it more natural and don't include any questions or sql queries.\n\n"
        f"Question: {state['question']}\n"
        f"SQL Query: {state['query']}\n"
        f"SQL Result: {state['result']}"
    )
    response = llm.invoke(prompt)
    return {"answer": response.content}


# Using langgraph to manage state transitions
from langgraph.graph import START, StateGraph

# Combine all steps into a state graph
graph_builder = StateGraph(State).add_sequence(
    [write_query, execute_query, generate_answer]
)
# Start from the initial state with the question
graph_builder.add_edge(START, "write_query")

# Compile the graph with memory checkpointing
graph = graph_builder.compile()

# List of questions to ask
questions = [
    "How many employees are there?",
    "Which track has sold the most?",
    "What is the name of the artist that has sold the most?",
    "Which genre has the most tracks?",
    "What is the longest track?"
]
for question in questions:
    final_result = graph.invoke({"question": question})
    print(final_result["answer"])
    print("*" * 10)