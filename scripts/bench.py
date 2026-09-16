"""Compile the engine, then search a small suite of positions to a fixed depth.

Prints compile time, which is the number the 90 second init budget cares about, and node
rate, which is the number everything else cares about.
"""

import os
import sys
import time

import chess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import nx_core as C  # noqa: E402
import nx_search as S  # noqa: E402

SUITE = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    "4rrk1/pp1n1pp1/2pb1q1p/3p4/3P1B2/2NBP2P/PPQ2PP1/R4RK1 w - - 0 17",
    "8/8/8/4k3/8/4K3/4P3/8 w - - 0 1",
    "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1",
]

DEPTH = int(sys.argv[1]) if len(sys.argv) > 1 else 10


def pv_text(c):
    return " ".join(C.move_to_uci(int(c.pv[0, i])) for i in range(int(c.pvlen[0])))


def main():
    t0 = time.perf_counter()
    c = S.make_ctx()
    alloc = time.perf_counter() - t0

    t0 = time.perf_counter()
    C.set_from_board(c.bb, c.sq, c.st, chess.Board())
    S.new_search(c)
    n = S.root_moves(c)
    S.score_moves(c, 0, n, 0)
    S.search_root(c, S.I4, -S.INF, S.INF, n)
    compile_s = time.perf_counter() - t0
    print(f"allocate {alloc:.2f}s   compile+warm {compile_s:.1f}s")

    total_nodes = 0
    total_time = 0.0
    for fen in SUITE:
        board = chess.Board(fen)
        C.set_from_board(c.bb, c.sq, c.st, board)
        S.new_search(c)
        c.ctl[S.C_GLEN] = 0
        n = S.root_moves(c)
        S.score_moves(c, 0, n, 0)
        t = time.perf_counter()
        score = 0
        for d in range(1, DEPTH + 1):
            score = S.search_root(c, d, -S.INF, S.INF, n)
        elapsed = time.perf_counter() - t
        nodes = int(c.ctl[S.C_NODES])
        total_nodes += nodes
        total_time += elapsed
        print(f"  d{DEPTH} {C.move_to_uci(int(c.ctl[S.C_BESTMOVE])):>6} "
              f"{score:>6}  {nodes:>10,}n {elapsed:6.2f}s "
              f"{nodes / max(elapsed, 1e-9) / 1000:7.0f} knps   pv: {pv_text(c)[:48]}")

    print(f"\ntotal {total_nodes:,} nodes in {total_time:.2f}s = "
          f"{total_nodes / total_time / 1e6:.2f} Mnps")


if __name__ == "__main__":
    main()
