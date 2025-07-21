# Transformers for sequences of chess moves

Work in progress.

From scratch pytorch implementation of Transformer models to predict sequences of chess moves. 
Current version is pure imitation learning (next-token prediction), with no notion of reward for 
good/valid moves.

Data: [Lichess Elite](https://database.nikonoel.fr/) database. 

Moves are encoded as sequences of three tokens [from, to, promotion]. 

Some sample games played [here](https://lichess.org/study/ZB0upGxH). 

