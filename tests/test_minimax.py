from src.core.board import Board
from src.agents.minmax import MinimaxAgent

def test_minimax_finds_winning_move():
    b = Board.empty()
    b.grid.setflags(write=True)

    # joueur -1 a déjà 3 pions en colonne 0 => le coup gagnant est col 0
    b.drop_piece_inplace(0, -1)
    b.drop_piece_inplace(0, -1)
    b.drop_piece_inplace(0, -1)

    agent = MinimaxAgent(depth=2, player=-1)
    action = agent.act(b)
    assert action == 0
