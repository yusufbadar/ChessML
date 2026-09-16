"""Exercise the interim search that covers a compilation overrun.

That path only runs on a machine slow enough to miss the init budget, which means it is
exactly the code least likely to be tested by playing games. So test it directly: force
it on, play a whole game with it, and check every move is legal.
"""

import sys
import time
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agent  # noqa: E402

agent.warm_done.wait()
if agent._warm_error:
    raise SystemExit(f"warm-up failed: {agent._warm_error[0]!r}")
if not agent.ready.is_set():
    raise SystemExit("full warm-up did not complete")

print("forcing the interim path (compiled primitives, interpreted recursion)")
agent.ready.clear()

board = chess.Board()
plies = 0
t0 = time.perf_counter()
while not board.is_game_over(claim_draw=True) and plies < 60:
    started = time.perf_counter()
    uci = agent.get_move(board.fen(), 60_000)
    spent = (time.perf_counter() - started) * 1000
    move = chess.Move.from_uci(uci)
    if move not in board.legal_moves:
        raise SystemExit(f"ILLEGAL {uci} in {board.fen()}")
    board.push(move)
    plies += 1
    if plies <= 6 or plies % 15 == 0:
        print(f"  ply {plies:>3} {uci}  {spent:6.0f}ms")
print(f"{plies} legal plies in {time.perf_counter() - t0:.1f}s, no illegal move")

# and a few awkward positions
agent.ready.clear()
for fen in ("4k3/PPP4p/8/2pP4/8/8/4p2P/4K3 w - c6 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "7k/8/8/8/8/8/6QQ/7K w - - 0 1",
            "8/8/8/8/8/5k2/6q1/7K b - - 0 1"):
    b = chess.Board(fen)
    uci = agent.get_move(fen, 30_000)
    ok = chess.Move.from_uci(uci) in b.legal_moves
    print(f"  {'ok ' if ok else 'ILLEGAL'} {uci}   {fen}")
    if not ok:
        raise SystemExit("illegal move from the interim search")

agent.ready.set()
print("interim search: all moves legal")
