"""

Inputs:
- train games
- test games


Outputs:
- train games perplexity
- test games perplexity
- number of illegal moves in top k=5, from different starting positions in the test set
- move quality over time
- mate in 1 stats

"""

from chess_seq import ChessGameEngine, load_model_from_hf
from chess_seq.evaluation.model_basic_eval import test_first_moves

model, config, encoder = load_model_from_hf("gamba_rossa")
test_first_moves(model, encoder, n_plies=30, prints=True)
