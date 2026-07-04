from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain import hub
from langchain.agents import create_tool_calling_agent, AgentExecutor
from dotenv import load_dotenv
import os

# Load config from .env
load_dotenv()

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# Weather information search
search = TavilySearchResults(max_results = 2, tavily_api_key=os.environ.get("TAVILY_API_KEY"))
# result = search.invoke("what is the weather in malmo, sweden")
# print(result)

# Load document from web
loader = WebBaseLoader("https://docs.smith.langchain.com/overview")
docs = loader.load()
documents = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
).split_documents(docs)
vector = FAISS.from_documents(documents, OllamaEmbeddings(model="llama3:8b"))
retriever = vector.as_retriever()
# print(retriever.invoke("how to upload a dataset")[0])
retriever_tool = create_retriever_tool(
    retriever,
    "langsmith_search",
    "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!",
)

# Define the tools
tools = [search, retriever_tool]

# define llm
model = init_chat_model("mistral:7b", model_provider="ollama")
# response = model.invoke([HumanMessage(content="hi!")])
# print(response.content)

# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/openai-functions-agent")
agent = create_tool_calling_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

msgs = ["Hi", "What's the weather in Almhult, Sweden?", "How can langsmith help with testing?"]
for m in msgs:
    print(f'Prompt: {m}')
    result = agent_executor.invoke({"input": m})
    print(f'Answer: {result['output']}')
    print('\n\n')