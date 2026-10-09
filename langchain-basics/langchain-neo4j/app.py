import os
from dotenv import load_dotenv
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph

load_dotenv()

# Load environment variables for Neo4j connection
GRAPH_URL = os.getenv("GRAPH_URL")
USER = os.getenv("USER")
PASSWORD = os.getenv("PASSWORD")

# Initialize Neo4jGraph with enhanced schema
graph = Neo4jGraph(url=GRAPH_URL, username=USER, password=PASSWORD, enhanced_schema=True)


from langchain_ollama import ChatOllama
llm = ChatOllama(
    model="qwen3:8b",
    temperature=0
)

# To be run at DB side to create a sample graph for testing. Uncomment to run.
# graph.query(
#     """
# MERGE (m:Movie {name:"Top Gun", runtime: 120})
# WITH m
# UNWIND ["Tom Cruise", "Val Kilmer", "Anthony Edwards", "Meg Ryan"] AS actor
# MERGE (a:Actor {name:actor})
# MERGE (a)-[:ACTED_IN]->(m)
# """
# )

# print(graph.schema)

# Create a GraphCypherQAChain instance
chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    verbose=True,
    allow_dangerous_requests=True
)

# result = chain.invoke({"query": "Which actors acted in the movie Top Gun?"})
result = chain.invoke({"query": "Why is PaymentService restarting after the deployment and what should I investigate?"})
print(result["result"])