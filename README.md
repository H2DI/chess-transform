# Transformers for sequences of chess moves

Pytorch (re-)implementation of Transformer models to predict sequences of chess moves. 
Current version is pure imitation learning (next-token prediction), with no notion of
 reward for good/valid moves.

### Model
Architecture is a reimplementation of QWEN3 0.6B. 

(~450M params with our vocab size)

Tokens: 'From x To x Promotion' (e.g. 'e2e4', 'f8g6', 'e7e8pq') 
(4608 possible tokens + 'start' + 'end' + 'pad')

### Data
[Lichess Elite](https://database.nikonoel.fr/) database. 
Total number of games: 12,421,396
Total number of tokens: 1,102,678,752 


### Results
Trained on an A100. 

Some sample games played against itself [here](https://lichess.org/study/ZbXAbPvL).

You might be able to play against the bot on [its lichess account](https://lichess.org/@/GambaRossa/all) .
Illegal moves are masked at inference.





### Future

Minor


Big
Improvement by self-play



