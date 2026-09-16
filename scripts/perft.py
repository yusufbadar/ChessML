"""Prove the move generator against the published perft reference positions.

Any disagreement here is a bug that would otherwise show up as a lost game, so this runs
before anything else does.
"""

import os
import sys
import time

import chess
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import nx_core as C  # noqa: E402

SUITE = [
    (chess.STARTING_FEN, [20, 400, 8902, 197281, 4865609, 119060324]),
    ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
     [48, 2039, 97862, 4085603, 193690690]),
    ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
     [14, 191, 2812, 43238, 674624, 11030083]),
    ("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
     [6, 264, 9467, 422333, 15833292]),
    ("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
     [44, 1486, 62379, 2103487, 89941194]),
    ("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
     [46, 2079, 89890, 3894594]),
]

MAX_NODES = int(sys.argv[1]) if len(sys.argv) > 1 else 6_000_000


def divide(fen, depth):
    """Per-move node counts, so a mismatch points at the move that causes it."""
    bb, sq, st = C.new_position()
    und = np.zeros((C.MAXPLY + 8, C.U_N), dtype=np.int64)
    ml = np.zeros((C.MAXPLY + 8, C.MAXMOVES), dtype=np.int32)
    C.set_from_board(bb, sq, st, chess.Board(fen))
    n = C.gen_moves(bb, sq, st, ml, np.int64(0), np.int64(0))
    out = {}
    for i in range(n):
        m = ml[0, i]
        C.do_move(bb, sq, st, und, np.int64(0), m)
        out[C.move_to_uci(m)] = C.perft(bb, sq, st, und, ml, np.int64(1), depth - 1)
        C.undo_move(bb, sq, st, und, np.int64(0))
    return out


def reference(fen, depth):
    board = chess.Board(fen)
    out = {}
    for move in board.legal_moves:
        board.push(move)
        out[move.uci()] = ref_count(board, depth - 1)
        board.pop()
    return out


def ref_count(board, depth):
    if depth == 0:
        return 1
    if depth == 1:
        return board.legal_moves.count()
    total = 0
    for move in board.legal_moves:
        board.push(move)
        total += ref_count(board, depth - 1)
        board.pop()
    return total


def main():
    bb, sq, st = C.new_position()
    und = np.zeros((C.MAXPLY + 8, C.U_N), dtype=np.int64)
    ml = np.zeros((C.MAXPLY + 8, C.MAXMOVES), dtype=np.int32)

    t0 = time.perf_counter()
    C.set_from_board(bb, sq, st, chess.Board())
    C.perft(bb, sq, st, und, ml, 0, 1)
    print(f"warm-up compile {time.perf_counter() - t0:.1f}s")

    failures = 0
    total_nodes = 0
    total_time = 0.0
    for fen, expected in SUITE:
        for depth, want in enumerate(expected, start=1):
            if want > MAX_NODES:
                continue
            C.set_from_board(bb, sq, st, chess.Board(fen))
            t = time.perf_counter()
            got = C.perft(bb, sq, st, und, ml, np.int64(0), depth)
            elapsed = time.perf_counter() - t
            total_nodes += got
            total_time += elapsed
            ok = "ok" if got == want else "MISMATCH"
            if got != want:
                failures += 1
            rate = got / elapsed / 1e6 if elapsed > 0 else 0.0
            print(f"  depth {depth}: {got:>12,} want {want:>12,}  {ok:>8}  "
                  f"{elapsed:6.2f}s {rate:6.2f} Mnps")
            if got != want:
                mine = divide(fen, depth)
                theirs = reference(fen, depth)
                for key in sorted(set(mine) | set(theirs)):
                    a, b = mine.get(key, "-"), theirs.get(key, "-")
                    if a != b:
                        print(f"      {key}: got {a} want {b}")
                break
        print(f"  {fen}")

    if total_time > 0:
        print(f"\n{total_nodes:,} nodes in {total_time:.2f}s = "
              f"{total_nodes / total_time / 1e6:.2f} Mnps")
    print("PERFT FAILURES:", failures)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
