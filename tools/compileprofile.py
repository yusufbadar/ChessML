"""True single-core compile cost, per jitted function, with no deadline.

tools/initprobe.py answers "does it fit". This answers "what is spending it".
"""

import ctypes
import os
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    ctypes.windll.kernel32.SetProcessAffinityMask(
        ctypes.windll.kernel32.GetCurrentProcess(), 1)
else:
    os.sched_setaffinity(0, {0})

os.environ.setdefault("NUMBA_OPT", "2")
os.environ.setdefault("NUMBA_SLP_VECTORIZE", "0")
os.environ.setdefault("NUMBA_LOOP_VECTORIZE", "0")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

T0 = time.monotonic()
import chess  # noqa: E402

t = time.monotonic()
import nx_core as C  # noqa: E402
t_core = time.monotonic() - t

t = time.monotonic()
import nx_eval as E  # noqa: E402
import nx_search as S  # noqa: E402
t_imports = time.monotonic() - t

ctx = S.make_ctx()
board = chess.Board()

stages = []


def stage(name, fn):
    t = time.monotonic()
    fn()
    stages.append((name, time.monotonic() - t))


stage("set_from_board (compute_key)",
      lambda: C.set_from_board(ctx.bb, ctx.sq, ctx.st, board))
stage("gen_moves", lambda: C.gen_moves(ctx.bb, ctx.sq, ctx.st, ctx.ml, S.I0, S.I0))
stage("evaluate", lambda: S.static_eval(ctx))
stage("see", lambda: C.see_value(ctx.bb, ctx.sq, ctx.st, ctx.ml[0, 0], ctx.gain))
stage("new_search", lambda: S.new_search(ctx))
stage("root_moves", lambda: S.root_moves(ctx))
n = S.root_moves(ctx)
stage("score_moves", lambda: S.score_moves(ctx, 0, n, 0))
stage("qsearch", lambda: S.qsearch(ctx, S.I0, -S.INF, S.INF))
stage("search_root d1", lambda: S.search_root(ctx, S.I1, -S.INF, S.INF, n))
stage("search_root d4 (null, lmr, history)",
      lambda: S.search_root(ctx, S.I4, -S.INF, S.INF, n))

total = time.monotonic() - T0
print(f"  import nx_core (incl. table build) {t_core:6.1f}s")
print(f"  import nx_eval + nx_search         {t_imports:6.1f}s")
for name, dt in stages:
    if dt > 0.05:
        print(f"  compile {name:<36} {dt:6.1f}s")
print(f"\n  TOTAL single-core warm-up          {total:6.1f}s  "
      f"(budget 90s, agent waits {85.0}s)")

dupes = []
for mod in (S, C, E):
    for name, fn in sorted(vars(mod).items()):
        sigs = getattr(fn, "signatures", None)
        if sigs and len(sigs) > 1:
            dupes.append(f"{name} x{len(sigs)} {sigs}")
print("  duplicate specialisations:", ", ".join(dupes) if dupes else "none")
