# Transformers for sequences of chess moves

Work in progress.

From scratch pytorch implementation of Transformer models to predict sequences of chess moves. 
Current version is pure imitation learning (next-token prediction), with no notion of reward for good/valid moves.

### Model
Architecture is a reimplementation of QWEN3 0.6B. 

(~450M params with our vocab size)

Tokens: 'From x To x Promotion' (e.g. 'e2e4', 'f8g6', 'e7e8pq') 
(4608 possible tokens + 'start' + 'end' + '')

### Data
[Lichess Elite](https://database.nikonoel.fr/) database. 
Total number of games: 12,421,396
Total number of tokens: 1,102,678,752 


### Results
Trained on an A100. 

Some sample games played [here](https://lichess.org/study/ZbXAbPvL).

### Future

Minor

Big
Improvement by self-play



