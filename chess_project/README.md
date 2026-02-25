# Chess learning

This project aims to train a neural network capable of determining whether a sequence of chess moves, expressed as a UCI string, is legally valid.

The model is trained on a large synthetic dataset composed of labeled sequences of up to 20 moves. Each sequence is classified as either legally consistent or containing an illegal move.

The central question explored in this project is whether a neural network can implicitly learn the rules of chess purely from labeled move sequences, without access to an explicit representation of the board state or a rule-based engine.
