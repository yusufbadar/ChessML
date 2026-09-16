"""Play a whole game against ourselves under the real clock and watch the time go.

The point is not the moves, it is the clock: does the agent finish a full-length game
inside 120s + 0.5s, and how bad is its worst single move. Flagging is the most common
self-inflicted loss and it does not show up in a four-game match at 20 seconds.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess  # noqa: E402

import agent  # noqa: E402

BASE_MS = 120_000
INCREMENT_MS = 500
PLY_CAP = 300


def main():
    start_fen = sys.argv[1] if len(sys.argv) > 1 else chess.STARTING_FEN
    board = chess.Board(start_fen)
    clock = {chess.WHITE: float(BASE_MS), chess.BLACK: float(BASE_MS)}
    worst = {chess.WHITE: 0.0, chess.BLACK: 0.0}
    spent = {chess.WHITE: 0.0, chess.BLACK: 0.0}

    for ply in range(PLY_CAP):
        if board.is_game_over(claim_draw=True):
            print(f"game over: {board.outcome(claim_draw=True).termination.name}")
            break
        mover = board.turn
        t0 = time.monotonic()
        uci = agent.get_move(board.fen(), int(clock[mover]))
        used = (time.monotonic() - t0) * 1000.0
        clock[mover] -= used
        spent[mover] += used
        worst[mover] = max(worst[mover], used)
        if clock[mover] < 0:
            print(f"FLAGGED on ply {ply} ({'white' if mover else 'black'})")
            return 1
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            print(f"ILLEGAL {uci} on ply {ply}")
            return 1
        board.push(move)
        clock[mover] += INCREMENT_MS
        if ply % 10 == 0:
            print(f"  ply {ply:>3} {uci} {used:7.0f}ms   "
                  f"white {clock[chess.WHITE] / 1000:6.1f}s  "
                  f"black {clock[chess.BLACK] / 1000:6.1f}s", flush=True)
    else:
        print("hit the 300 ply cap")

    for colour, name in ((chess.WHITE, "white"), (chess.BLACK, "black")):
        print(f"{name}: {spent[colour] / 1000:.1f}s spent, "
              f"{clock[colour] / 1000:.1f}s left, worst move {worst[colour]:.0f}ms")
    print(f"{board.fullmove_number} moves played")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
