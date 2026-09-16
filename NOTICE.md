# Third-party code and data

## `harness/` and `baselines/` — AI Chessathon starter kit

These two directories are carried **unmodified** from the competition's starter
repository. `harness/` is the platform's own referee, clock and wire protocol; keeping it
byte-identical is what makes local games mean anything. `baselines/` are the reference
agents shipped for comparison.

Everything else in this repository — `agent.py`, `nx_*.py`, `scripts/`, `tools/` — is
original work.

> MIT License
>
> Copyright (c) 2026 Advit Arora
>
> Permission is hereby granted, free of charge, to any person obtaining a copy of this
> software and associated documentation files (the "Software"), to deal in the Software
> without restriction, including without limitation the rights to use, copy, modify,
> merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
> permit persons to whom the Software is furnished to do so, subject to the following
> conditions:
>
> The above copyright notice and this permission notice shall be included in all copies
> or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
> INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
> PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
> HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
> CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
> OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Training data — Lichess open database

The evaluation weights in `nx_params.py` were fitted to positions from the
[Lichess evaluation database](https://database.lichess.org/#evals): 394 million positions
annotated by Stockfish, released under **CC0** (public domain dedication). The tactics
regression suite uses the [Lichess puzzle database](https://database.lichess.org/#puzzles),
also CC0.

Neither dataset is redistributed here. `scripts/fetch_evals.py` and `scripts/puzzles.py`
contain the exact code that downloaded and filtered them, so the provenance of every
number in `nx_params.py` can be checked rather than taken on trust.

No chess engine's source code is used, vendored, linked, or executed anywhere in this
repository. See the "On using Stockfish-labelled data" section of the README for why
training on engine-annotated positions was permitted under the competition rules.

## `docs/banner.jpg` — event artwork

The banner is the official promotional artwork for the Optiver Chess x Machine Learning
Hackathon (AI Chessathon 2026), copyright its respective owners. It is included solely to
identify the competition this project was built for. No affiliation with, sponsorship by,
or endorsement from Optiver or the event organisers is implied.

## Runtime dependencies

`python-chess`, `numpy` and `numba`, all installed from PyPI and not vendored.
