from typing import List, Optional
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model

# Pydantic model
# Single person
class Person(BaseModel):
    """Information about a person."""
    name: Optional[str] = Field(default=None, description="The name of the person")
    hair_color: Optional[str] = Field(
        default=None, description="The color of the person's hair"
    )
    height_in_meters: Optional[float] = Field(
        default=None,
        description="Height in meters as a float number. Convert any feet or inches to meters (1 foot = 0.3048 meters, e.g. 6 feet is ~1.83 meters).",
    )

# List of persons
class Data(BaseModel):
    """Extracted data about people."""
    people: List[Person] = Field(
        description="List of all people mentioned in the text"
    )

# Define a custom prompt to provide instructions and any additional context.
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert extraction algorithm. "
            "Extract name, hair color, and height for each person. "
            "Always convert height from feet to meters as a float (1 foot = 0.3048 meters). "
            "If you do not know the value of an attribute, return null for that attribute.",
        ),
        ("human", "{text}"),
    ]
)

def print_formatted(title: str, obj: Person | Data) -> None:
    print(f"\n{'=' * 15} {title} {'=' * 15}")
    # Formatted JSON output
    print("\n[JSON Output]")
    print(obj.model_dump_json(indent=2))

# define model
llm = init_chat_model("qwen3:8b", model_provider="ollama")

# 1. Single Person Extraction
structured_llm_single = llm.with_structured_output(schema=Person)
text1 = "Alan Smith is 6 feet tall height and has blond hair color."
prompt1 = prompt_template.invoke({"text": text1})
result1 = structured_llm_single.invoke(prompt1)
print_formatted("Single Person Extraction", result1)

# 2. Multiple People Extraction
structured_llm_multi = llm.with_structured_output(schema=Data)
text2 = "My name is Jeff, my hair is red and i am 4.5 feet tall height. Anna has the same hair color as Jeff.Projjal is 170cm tall and has black hair."
prompt2 = prompt_template.invoke({"text": text2})
result2 = structured_llm_multi.invoke(prompt2)
print_formatted("Multiple People Extraction", result2)