"""Total warm-up cost and resulting node rate, for one numba configuration."""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess  # noqa: E402

t0 = time.perf_counter()
import nx_core as C  # noqa: E402
import nx_search as S  # noqa: E402

c = S.make_ctx()
C.set_from_board(c.bb, c.sq, c.st, chess.Board())
S.new_search(c)
n = S.root_moves(c)
S.score_moves(c, 0, n, 0)
S.search_root(c, 4, -S.INF, S.INF, n)
warm = time.perf_counter() - t0

nodes = 0
elapsed = 0.0
for fen in ["r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "4rrk1/pp1n1pp1/2pb1q1p/3p4/3P1B2/2NBP2P/PPQ2PP1/R4RK1 w - - 0 17",
            "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10"]:
    C.set_from_board(c.bb, c.sq, c.st, chess.Board(fen))
    S.new_search(c)
    c.tt[:] = 0
    n = S.root_moves(c)
    S.score_moves(c, 0, n, 0)
    t = time.perf_counter()
    for d in range(1, 12):
        S.search_root(c, d, -S.INF, S.INF, n)
    elapsed += time.perf_counter() - t
    nodes += int(c.ctl[S.C_NODES])

tag = " ".join(f"{k.replace('NUMBA_', '')}={v}" for k, v in sorted(os.environ.items())
               if k.startswith("NUMBA_"))
print(f"{tag or 'defaults':<52} warm {warm:6.1f}s   d11 {nodes:>9,}n "
      f"{nodes / elapsed / 1e6:5.2f} Mnps")
