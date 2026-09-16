import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chess
import nx_core as C
import nx_search as S

c = S.make_ctx()
C.set_from_board(c.bb, c.sq, c.st, chess.Board(
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"))
S.bench_eval(c, 1); S.bench_movegen(c, 1)
for name, fn, reps in (("eval (cold pawn hash)", S.bench_eval, 300000),
                       
                       ("gen_moves", S.bench_movegen, 300000)):
    t = time.perf_counter(); fn(c, reps); dt = (time.perf_counter()-t)/reps*1e9
    print(f"  {name:<24} {dt:7.0f} ns  ({1e9/dt/1e6:6.2f} M/s)")
