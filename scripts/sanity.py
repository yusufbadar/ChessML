"""Is the evaluation sane, and does the root agree with a plain search of each move."""

import os
import sys

import chess
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import nx_core as C  # noqa: E402
import nx_search as S  # noqa: E402

EVAL_CHECKS = [
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "start, white to move"),
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1", "start, black to move"),
    ("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3", "italian, black"),
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/1NBQKBNR w Kkq - 0 1", "white minus a rook"),
    ("rnbqkbn1/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQq - 0 1", "black minus a rook"),
    ("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1", "K+P vs K"),
    ("4k3/8/8/8/8/8/8/4K3 w - - 0 1", "bare kings"),
    ("6k1/5ppp/8/8/8/8/5PPP/6K1 w - - 0 1", "symmetric pawns"),
]


def main():
    c = S.make_ctx()
    for fen, label in EVAL_CHECKS:
        C.set_from_board(c.bb, c.sq, c.st, chess.Board(fen))
        print(f"  eval {int(S.static_eval(c)):>7}   {label}")

    fen = sys.argv[1] if len(sys.argv) > 1 else EVAL_CHECKS[2][0]
    depth = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    board = chess.Board(fen)
    print(f"\nroot scores at depth {depth}: {fen}")

    scored = []
    for move in board.legal_moves:
        board.push(move)
        C.set_from_board(c.bb, c.sq, c.st, board)
        S.new_search(c)
        c.tt[:] = 0
        c.gkeys[0] = c.st[C.ST_KEY]
        c.ctl[S.C_GLEN] = 0
        n = S.root_moves(c)
        S.score_moves(c, 0, n, 0)
        v = 0
        for d in range(1, depth):
            v = S.search_root(c, d, -S.INF, S.INF, n)
        board.pop()
        scored.append((-v, move.uci()))
    scored.sort(reverse=True)
    for v, uci in scored[:8]:
        print(f"    {uci:>6} {int(v):>7}")

    C.set_from_board(c.bb, c.sq, c.st, board)
    S.new_search(c)
    c.tt[:] = 0
    c.gkeys[0] = c.st[C.ST_KEY]
    c.ctl[S.C_GLEN] = 0
    n = S.root_moves(c)
    S.score_moves(c, 0, n, 0)
    for d in range(1, depth + 1):
        v = S.search_root(c, d, -S.INF, S.INF, n)
        pv = " ".join(C.move_to_uci(int(c.pv[0, i])) for i in range(int(c.pvlen[0])))
        print(f"    root d{d:<2} {int(v):>7}  {C.move_to_uci(int(c.ctl[S.C_BESTMOVE])):>6}"
              f"  {int(c.ctl[S.C_NODES]):>9}n  pv {pv}")


if __name__ == "__main__":
    main()
