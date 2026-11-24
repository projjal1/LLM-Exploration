from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model

load_dotenv()

# max_tokens=280 physically caps the output to ~200 words
llm = init_chat_model("qwen3:8b", model_provider="ollama", max_tokens=280)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert psychologist. Write an essay under 200 words."),
    ("user", "Write an essay on human psychology.")
])

chain = prompt | llm | StrOutputParser()

print("Streaming essay response:\n")
for chunk in chain.stream({}):
    print(chunk, end="", flush=True)
print()