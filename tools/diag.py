"""Where does compile time go, and where does search time go."""

import os
import sys
import time

opt = os.environ.get("NUMBA_OPT", "default")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess  # noqa: E402
import numpy as np  # noqa: E402

t = time.perf_counter()
import nx_core as C  # noqa: E402
t_core_import = time.perf_counter() - t

t = time.perf_counter()
import nx_search as S  # noqa: E402
t_search_import = time.perf_counter() - t

c = S.make_ctx()
board = chess.Board("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
C.set_from_board(c.bb, c.sq, c.st, board)

stages = []


def stage(name, fn):
    t = time.perf_counter()
    fn()
    stages.append((name, time.perf_counter() - t))


stage("movegen+perft", lambda: C.perft(c.bb, c.sq, c.st, c.und, c.ml, 0, 1))
stage("see", lambda: C.see_value(c.bb, c.sq, c.st, c.ml[0, 0], c.gain))
stage("gives_check", lambda: C.gives_check(c.bb, c.sq, c.st, c.ml[0, 0]))
stage("evaluate", lambda: S.static_eval(c))
stage("new_search", lambda: S.new_search(c))
stage("root_moves", lambda: S.root_moves(c))
n = S.root_moves(c)
stage("score_moves", lambda: S.score_moves(c, 0, n, 0))
stage("qsearch", lambda: S.qsearch(c, 0, -S.INF, S.INF))
stage("negamax d1", lambda: S.search_root(c, 1, -S.INF, S.INF, n))
stage("negamax d4", lambda: S.search_root(c, 4, -S.INF, S.INF, n))

print(f"NUMBA_OPT={opt}")
print(f"  import nx_core   {t_core_import:6.1f}s   (table build)")
print(f"  import nx_search {t_search_import:6.1f}s")
for name, dt in stages:
    print(f"  compile {name:<16} {dt:6.1f}s")
print(f"  TOTAL {t_core_import + t_search_import + sum(d for _, d in stages):.1f}s")

# ---- runtime micro-benchmarks -----------------------------------------------------------
reps = 200_000
t = time.perf_counter()
for _ in range(reps // 1000):
    for _ in range(1000):
        S.static_eval(c)
ev = (time.perf_counter() - t) / reps * 1e9
print(f"\nevaluate: {ev:.0f} ns/call  ({1e9 / ev / 1e6:.2f} M/s)")

t = time.perf_counter()
for _ in range(reps // 100):
    C.gen_moves(c.bb, c.sq, c.st, c.ml, 0, False)
gm = (time.perf_counter() - t) / (reps // 100) * 1e9
print(f"gen_moves: {gm:.0f} ns/call")

for fen in ["r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "4rrk1/pp1n1pp1/2pb1q1p/3p4/3P1B2/2NBP2P/PPQ2PP1/R4RK1 w - - 0 17"]:
    C.set_from_board(c.bb, c.sq, c.st, chess.Board(fen))
    S.new_search(c)
    c.tt[:] = 0
    n = S.root_moves(c)
    S.score_moves(c, 0, n, 0)
    t = time.perf_counter()
    for d in range(1, 11):
        S.search_root(c, d, -S.INF, S.INF, n)
    el = time.perf_counter() - t
    print(f"d10 search: {int(c.ctl[S.C_NODES]):>10,} nodes {el:5.2f}s "
          f"{int(c.ctl[S.C_NODES]) / el / 1e6:.2f} Mnps")
