import random

import chess
import eel
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

# Initialize Eel
eel.init('web')

# Initialize LangChain chat model via the modern helper
model = init_chat_model("mistral:7b", model_provider="ollama")

system_template = (
    "You are a professional chess player. You will be given a FEN position and a numbered list "
    "of legal moves in SAN notation. Reply with ONLY the number of the single best move and nothing else."
)

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "FEN: {board_fen}\nLegal moves:\n{legal_moves}")]
)


def get_ai_move(board_fen: str) -> str:
    board = chess.Board(board_fen)
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return board_fen  # checkmate or stalemate: nothing left to play

    san_moves = [board.san(move) for move in legal_moves]
    numbered = "\n".join(f"{i + 1}. {san}" for i, san in enumerate(san_moves))

    prompt = prompt_template.invoke({"board_fen": board_fen, "legal_moves": numbered})
    response = model.invoke(prompt)
    content = getattr(response, "content", str(response)).strip()
    print(f"AI Response: {content}")

    choice = next((int(tok) for tok in content.split() if tok.isdigit()), None)
    if choice is None or not (1 <= choice <= len(legal_moves)):
        move = random.choice(legal_moves)  # fallback if the model didn't return a valid choice
    else:
        move = legal_moves[choice - 1]

    board.push(move)
    return board.fen()


@eel.expose
def suggest_move(fen_str: str) -> str:
    return get_ai_move(fen_str)


if __name__ == "__main__":
    eel.start('index.html', size=(600, 600))