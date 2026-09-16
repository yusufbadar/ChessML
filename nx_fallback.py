"""A small pure-python search, used only if the compiled engine is not ready in time.

The compiled engine needs about forty seconds of numba compilation and the platform
allows ninety, so on any machine like the one this was developed on the fallback never
runs. It exists because "not ready yet" must not mean "no move": a slower judge machine
would otherwise cost us a game before the engine ever plays one.

It is an ordinary alpha-beta with material, piece-square tables, quiescence and MVV-LVA
ordering. Perhaps 1800 strength, which is enough to hold a position together for the two
or three moves the real engine might need to finish compiling.
"""

import time

import chess

VALUE = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 335, chess.ROOK: 500,
         chess.QUEEN: 950, chess.KING: 0}

# Centralisation for the pieces that want it, advancement for pawns. Built, not tabulated.
PSQT = {}
for _pt, _weight in ((chess.PAWN, 0), (chess.KNIGHT, 30), (chess.BISHOP, 16),
                     (chess.ROOK, 6), (chess.QUEEN, 8), (chess.KING, -22)):
    _t = [0] * 64
    for _sq in range(64):
        _f, _r = _sq & 7, _sq >> 3
        _centre = (1.0 - abs(_f - 3.5) / 3.5) * (1.0 - abs(_r - 3.5) / 3.5)
        if _pt == chess.PAWN:
            _t[_sq] = int((0, 0, 4, 12, 26, 48, 80, 0)[_r]
                          + (-5, -2, 2, 8, 8, 2, -2, -5)[_f] * (1 if _r < 4 else 0))
        else:
            _t[_sq] = int(_weight * (2 * _centre - 1))
    PSQT[_pt] = _t

MATE = 30000


def evaluate(board):
    score = 0
    for square, piece in board.piece_map().items():
        v = VALUE[piece.piece_type] + PSQT[piece.piece_type][
            square if piece.color == chess.WHITE else square ^ 56]
        score += v if piece.color == chess.WHITE else -v
    return score if board.turn == chess.WHITE else -score


def _order(board, moves, tt_move):
    def key(move):
        if move == tt_move:
            return 1_000_000
        if board.is_capture(move):
            victim = board.piece_type_at(move.to_square) or chess.PAWN
            attacker = board.piece_type_at(move.from_square) or chess.PAWN
            return 10_000 + VALUE[victim] * 8 - VALUE[attacker]
        return 0
    return sorted(moves, key=key, reverse=True)


def _quiesce(board, alpha, beta, deadline):
    stand = evaluate(board)
    if stand >= beta:
        return beta
    alpha = max(alpha, stand)
    for move in _order(board, [m for m in board.legal_moves if board.is_capture(m)], None):
        if time.monotonic() > deadline:
            return alpha
        board.push(move)
        score = -_quiesce(board, -beta, -alpha, deadline)
        board.pop()
        if score >= beta:
            return beta
        alpha = max(alpha, score)
    return alpha


def _search(board, depth, alpha, beta, deadline, table, ply):
    if time.monotonic() > deadline:
        return alpha
    if depth <= 0:
        return _quiesce(board, alpha, beta, deadline)
    key = board._transposition_key()
    entry = table.get(key)
    tt_move = entry[2] if entry else None
    if entry and entry[0] >= depth:
        return entry[1]
    moves = list(board.legal_moves)
    if not moves:
        return -MATE + ply if board.is_check() else 0
    best = -MATE * 2
    best_move = None
    for move in _order(board, moves, tt_move):
        board.push(move)
        score = -_search(board, depth - 1, -beta, -alpha, deadline, table, ply + 1)
        board.pop()
        if score > best:
            best, best_move = score, move
        alpha = max(alpha, score)
        if alpha >= beta:
            break
    table[key] = (depth, best, best_move)
    return best


def choose(board, budget_ms):
    """Best move found inside the budget, or the first legal move if there is no time."""
    deadline = time.monotonic() + budget_ms / 1000.0
    moves = list(board.legal_moves)
    if not moves:
        return None
    best = moves[0]
    table = {}
    for depth in range(1, 32):
        alpha, best_here = -MATE * 2, None
        for move in _order(board, moves, best):
            board.push(move)
            score = -_search(board, depth - 1, -MATE * 2, -alpha, deadline, table, 1)
            board.pop()
            if best_here is None or score > alpha:
                alpha, best_here = score, move
            if time.monotonic() > deadline:
                break
        if best_here is not None and time.monotonic() <= deadline:
            best = best_here
        if time.monotonic() > deadline:
            break
    return best
