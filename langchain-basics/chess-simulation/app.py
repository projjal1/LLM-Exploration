import eel
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate

# Initialize Eel
eel.init('web')

# Initialize LangChain with Ollama
llm = Ollama(model='llama3:8b')

# template for prompt
prompt_template= """
    You are a chess AI playing a simulated match.
    The current board state is given in FEN: "{board_fen}".
    Determine whose turn it is from the FEN (the second field: 'w' means white's turn, 'b' means black's turn).
    Make the best legal move for the opposite player whose turn it is and return only one FEN string.
    Execute the task and return updated FEN string as per line
    FEN: <fen_value>
    Don't include any reasoning other than fen value, no commentary, no explanations in response.
    """
prompt = PromptTemplate(
    input_variables= ["board_fen"],
    template=prompt_template
)
chain = prompt | llm

def get_ai_move(board_fen):
    response = chain.invoke(input = {"board_fen": board_fen}).strip()
    return response

@eel.expose
def suggest_move(fen_str):
    value = str(get_ai_move(fen_str))
    value = value.splitlines()
    move = value[0].replace("FEN: ", "")
    return move

# Start Eel
eel.start('index.html', size=(600, 600))