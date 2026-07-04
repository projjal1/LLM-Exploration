# Create server parameters for stdio connection
from mcp import ClientSession
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama
import asyncio
from mcp.client.streamable_http import streamablehttp_client

# Use Ollama's Llama 3 (8B)
model = ChatOllama(
    model="mistral:7b",   # Model name for Ollama
    temperature=0,       # Optional: set low temperature for deterministic output
)

abs_path = "/Users/projjalgop/Documents/Code-Artifacts/ai-codes/mcp-flight-booking/"

MCP_SERVER_URL = "http://127.0.0.1:8000/mcp"

async def run_agent():
    async with streamablehttp_client(MCP_SERVER_URL) as (
        read_stream,
        write_stream,
        _,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            # Initialize the connection
            await session.initialize()

            # Get tools
            tools = await load_mcp_tools(session)

            # Create and run the agent
            agent = create_react_agent(model, tools)

            # Uncomment to run agents one at a time
            
            agent_response = await agent.ainvoke({"messages": "What is the airport code for Kolkata? Fetch data from ABC Travel Agency only."})
            print(agent_response['messages'][-1].content)

            # agent_response = await agent.ainvoke({"messages": "Show me flight fares for routes between Copenhagen as departure airport and Kolkata as arrival airport. Fetch data from ABC Travel Agency only."})
            # print(agent_response['messages'][-1].content)

            # agent_response = await agent.ainvoke({"messages": "I want to know information about flight EK 569. Fetch data from ABC Travel Agency only."})
            # print(agent_response['messages'][-1].content)

# Run the async function
if __name__ == "__main__":
    asyncio.run(run_agent())