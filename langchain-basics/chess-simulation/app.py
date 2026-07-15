import random

import chess
import eel
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

# Initialize Eel
eel.init('web')

# Initialize LangChain chat model via the modern helper
model = init_chat_model("qwen2.5:7b-instruct", model_provider="ollama")

system_template = (
    "You are a chess engine. Pick the strongest legal move for the side to move. "
    "Reply with ONLY its number - no words, no punctuation."
)

user_template = (
    "{turn} to move.\n\n"
    "{board_ascii}\n\n"
    "Legal moves:\n{legal_moves}"
)

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", user_template)]
)


def get_ai_move(board_fen: str) -> str:
    board = chess.Board(board_fen)
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return board_fen  # checkmate or stalemate: nothing left to play

    turn = "White" if board.turn else "Black"
    san_moves = [board.san(move) for move in legal_moves]
    numbered = "\n".join(f"{i + 1}. {san}" for i, san in enumerate(san_moves))

    prompt = prompt_template.invoke(
        {"turn": turn, "board_ascii": str(board), "legal_moves": numbered}
    )
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