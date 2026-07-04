var board = Chessboard('board', 'start');
var game = new Chess();

async function getAIMove() {
  var fen = game.fen();
  await eel.suggest_move(fen)(function(move) {
    if (game.move(move, { sloppy: true })) {
      board.position(game.fen());
    } else {
      alert("Invalid move suggested: " + move);
    }
  });
}