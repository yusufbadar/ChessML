"""Fast in-process A/B match between two evaluation weight vectors.

Games are played to a node budget rather than a clock, so a result means the same thing
on a throttled laptop as on a server and two runs of the same match are identical. This
is the tool that decides whether a change is real: a tuning run that lowers the fitting
error but loses the match did not help.

    python scripts/arena.py --a data/tuned.npy --games 120 --nodes 6000
"""

import argparse
import math
from pathlib import Path

import chess
import numpy as np

from engine_api import search, search_position, to_move, warm

import nx_params as params

OPENINGS = [
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
    "rnbqkb1r/pppppppp/5n2/8/2PP4/8/PP2PPPP/RNBQKBNR b KQkq - 0 2",
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    "rnbqkb1r/pp2pppp/3p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R b KQkq - 2 5",
    "rnbqkb1r/ppp1pppp/5n2/3p4/2PP4/5N2/PP2PPPP/RNBQKB1R b KQkq - 1 3",
    "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    "rnbqkbnr/pp2pppp/8/2pp4/3P4/2N2N2/PPP1PPPP/R1BQKB1R b KQkq - 1 3",
    "rnbqk2r/ppppppbp/5np1/8/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 2 4",
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 b kq - 5 4",
    "rnbqkb1r/pp3ppp/4pn2/2pp4/2PP4/5NP1/PP2PP1P/RNBQKB1R w KQkq - 0 5",
    "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQK2R b KQkq - 0 5",
    "rnbqkbnr/ppp2ppp/8/3pp3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 3",
]

HELDOUT_LINES = (
    "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6",
    "d2d4 d7d5 c2c4 e7e6 b1c3 g8f6",
    "c2c4 e7e5 b1c3 g8f6 g2g3 d7d5",
    "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4",
    "e2e4 c7c6 d2d4 d7d5 b1c3",
    "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7",
    "g1f3 d7d5 g2g3 g8f6 f1g2",
    "c2c4 c7c5 b1c3 b8c6 g2g3 g7g6",
    "d2d4 f7f5 g2g3 g8f6 f1g2",
    "e2e4 e7e6 d2d4 d7d5 b1c3",
    "e2e4 c7c5 b1c3 b8c6 f2f4",
    "d2d4 g8f6 c2c4 e7e6 g1f3 d7d5",
)

PIECE_POINTS = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5,
                chess.QUEEN: 9}


def heldout_openings():
    result = []
    for line in HELDOUT_LINES:
        board = chess.Board()
        for uci in line.split():
            board.push_uci(uci)
        result.append(board.fen())
    return result


def adjudicate(board):
    """The referee's rule: 300 plies without a result goes to material."""
    balance = sum(v * (len(board.pieces(p, chess.WHITE)) - len(board.pieces(p, chess.BLACK)))
                  for p, v in PIECE_POINTS.items())
    return 1.0 if balance > 0 else (0.0 if balance < 0 else 0.5)


def play_game(white, black, fen, nodes):
    board = chess.Board(fen)
    for ply in range(300):
        if board.is_game_over(claim_draw=True):
            outcome = board.outcome(claim_draw=True)
            if outcome.winner is None:
                return 0.5
            return 1.0 if outcome.winner == chess.WHITE else 0.0
        ctx = white if board.turn == chess.WHITE else black
        uci, _, _ = search_position(ctx, board, nodes)
        move = to_move(board, uci)
        if move is None:
            return 0.0 if board.turn == chess.WHITE else 1.0
        board.push(move)
    return adjudicate(board)


def reset_for_game(ctx):
    """Remove cross-game learning so paired colour swaps are exactly comparable."""
    contempt = ctx.ctl[search.C_CONTEMPT]
    ctx.tt[:] = 0
    ctx.ptt[:] = 0
    ctx.killers[:] = 0
    ctx.hist[:] = 0
    ctx.conth[:] = 0
    ctx.cmove[:] = 0
    ctx.ctl[:] = 0
    ctx.ctl[search.C_CONTEMPT] = contempt


def elo(score, games):
    if score <= 0 or score >= games:
        return float("inf") * (1 if score > 0 else -1)
    p = score / games
    return -400.0 * math.log10(1.0 / p - 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", type=Path, help="weights for side A (default: built-in)")
    parser.add_argument("--b", type=Path, help="weights for side B (default: built-in)")
    parser.add_argument("--contempt-a", type=int, default=None)
    parser.add_argument("--contempt-b", type=int, default=None)
    parser.add_argument("--games", type=int, default=60)
    parser.add_argument("--nodes", type=int, default=6000)
    parser.add_argument("--suite", choices=("train", "heldout", "all"), default="all")
    args = parser.parse_args()

    def weights(path):
        return np.load(path).astype(np.int32) if path else params.load().astype(np.int32)

    a = search.make_ctx(weights(args.a))
    b = search.make_ctx(weights(args.b))
    if args.contempt_a is not None:
        a.ctl[search.C_CONTEMPT] = args.contempt_a
    if args.contempt_b is not None:
        b.ctl[search.C_CONTEMPT] = args.contempt_b
    warm(a)
    print(f"A = {args.a or 'built-in'}   B = {args.b or 'built-in'}   "
          f"{args.games} paired games at {args.nodes} nodes", flush=True)

    score = 0.0
    wins = draws = losses = 0
    heldout = heldout_openings()
    openings = OPENINGS if args.suite == "train" else (
        heldout if args.suite == "heldout" else OPENINGS + heldout)
    for game in range(args.games):
        fen = openings[(game // 2) % len(openings)]
        reset_for_game(a)
        reset_for_game(b)
        a_white = game % 2 == 0
        result = play_game(a if a_white else b, b if a_white else a, fen, args.nodes)
        point = result if a_white else 1.0 - result
        score += point
        wins += point == 1.0
        draws += point == 0.5
        losses += point == 0.0
        if (game + 1) % 10 == 0:
            print(f"  {game + 1:>4} games: {score:>6.1f} "
                  f"({100 * score / (game + 1):.1f}%)  +{wins} ={draws} -{losses}",
                  flush=True)

    played = args.games
    margin = 1.96 * math.sqrt(max(1e-9, (score / played) * (1 - score / played) / played))
    print(f"\nA scores {score}/{played} = {100 * score / played:.1f}%  "
          f"+{wins} ={draws} -{losses}")
    print(f"elo {elo(score, played):+.0f}  "
          f"(95% band {elo(max(0.01, score - margin * played), played):+.0f} .. "
          f"{elo(min(played - 0.01, score + margin * played), played):+.0f})")


if __name__ == "__main__":
    main()
