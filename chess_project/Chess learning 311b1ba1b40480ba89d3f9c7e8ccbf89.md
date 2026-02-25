# Chess learning

This project aims to train a neural network capable of determining whether a sequence of chess moves, expressed as a UCI string, is legally valid.

The model is trained on a large synthetic dataset composed of labeled sequences of up to 20 moves. Each sequence is classified as either legally consistent or containing an illegal move.

The central question explored in this project is whether a neural network can implicitly learn the rules of chess purely from labeled move sequences, without access to an explicit representation of the board state or a rule-based engine.

This project heavily relies on the `python-chess` library, a powerful and elegant toolkit that makes it possible to programmatically simulate games, validate moves, and generate synthetic datasets.

Thanks to this library, we can efficiently construct legal and illegal move sequences without implementing the rules of chess from scratch.

We begin with the necessary imports:

```python
import chess
import chess.svg
import random
import itertools
import numpy as np
import pandas as pd
```

The first step of this project is to generate the dataset.

To do so, we need a function capable of producing sequences of legal chess moves.

```python
def generate_legal_random_sequence(n_moves=10):
    # Initialize a standard chess board in the starting position
    board = chess.Board()
    
    # List to store the sequence of moves in UCI format
    moves = []

    # Generate up to n_moves legal moves
    for _ in range(n_moves):
        
        # Stop early if the game has ended (checkmate, stalemate, etc.)
        if board.is_game_over():
            break

        # Randomly select one move from the current list of legal moves
        move = random.choice(list(board.legal_moves))
        
        # Apply the move to update the board state
        board.push(move)
        
        # Store the move in UCI string format (e.g., "e2e4")
        moves.append(move.uci())

    # Return the full sequence as a single string separated by "-"
    return '-'.join(moves)
```

Let’s see an example:

```python
generate_legal_random_sequence(5)
'g1f3-c7c6-f3d4-d8a5-d4c6'
```

We then need a function capable of generating a full move sequence.

```python
def push_sequence_of_moves(moves):
    # Initialize a new chess board in the standard starting position
    board = chess.Board()
    
    # If the input string is empty, return the initial board
    if moves == "":
        return board
    
    # Split the sequence into individual UCI moves
    for mv in moves.split('-'):
        
        # Convert the UCI string (e.g., "e2e4") into a Move object
        move_obj = chess.Move.from_uci(mv)
        
        # Check whether the move is legal in the current board state
        # If not, the sequence is invalid and we return None
        if move_obj not in board.legal_moves:
            return None
        
        # Apply the legal move to update the board state
        board.push(move_obj)
    
    # If all moves are legal, return the final board state
    return board
```

We can even visualise the generated sequences:

```python
# Generate a random legal sequence of 3 moves
test_sequence = generate_legal_random_sequence(3)

# Reconstruct the final board position by applying the sequence
final_board = push_sequence_of_moves(test_sequence)

# Print the generated sequence in UCI format
print(test_sequence)

# Display the final board position
display(final_board)
```

![image.png](Chess%20learning/image.png)

We now turn to a probabilistic perspective.
First, we define a function that generates completely random move sequences.

```python
def generate_random_sequence(n_moves=10):
    # Define board file letters (columns a–h)
    letters = ['a','b','c','d','e','f','g','h']
    
    # Define board rank numbers (rows 1–8)
    numbers = list(range(1,9))
    
    # Generate all possible board squares (e.g., "a1", "e4", etc.)
    positions = list(
        map(
            lambda el: el[0] + str(el[1]),
            list(itertools.product(letters, numbers))
        )
    )
    
    # List to store the generated moves
    moves = []
    
    # Generate n_moves completely random moves (no legality checks)
    for _ in range(n_moves):
        
        # Random starting square
        start = random.choice(positions)
        
        # Random destination square different from the starting square
        stop = random.choice(list(set(positions).difference({start})))
        
        # Construct UCI-like move string (e.g., "a2a5")
        mv = start + stop
        
        moves.append(mv)
    
    # Return the full sequence as a single string separated by "-"
    return '-'.join(moves)
```

Example:

```python
generate_random_sequence(5)
'f2h8-f2c6-d2b2-a2d5-g4d1'
```

Such a string is meaningless from a chess perspective.
Below, we define a function that analyzes an invalid sequence and identifies the exact move at which it becomes illegal.

```python
def analyze_illegal_sequence(sequence):
    # Initialize a new chess board in the standard starting position
    board = chess.Board()
    
    # Split the sequence into individual UCI moves
    moves = sequence.split('-')
    
    # Iterate over the moves while keeping track of the index
    for i, mv in enumerate(moves):
        
        # Convert the UCI string into a Move object
        move_obj = chess.Move.from_uci(mv)
        
        # If the move is not legal in the current position,
        # return diagnostic information
        if move_obj not in board.legal_moves:
            return {
                "illegal_move_index": i,          # Position of the first illegal move
                "illegal_move": mv,               # The illegal move itself
                "board_before_error": board.copy()  # Board state just before the error
            }
        
        # Otherwise, apply the move and continue
        board.push(move_obj)
    
    # If no illegal move is found, return None
    return None
```

This function allows us to precisely identify the point at which a random sequence violates the rules of chess, making it easier to visualize and analyze incorrect predictions produced by the neural network.

For instance, consider the last randomly generated sequence:

```python
# Convert the illegal move (in UCI format) into a Move object
move_obj = chess.Move.from_uci(
    analyze_illegal_sequence('f2h8-f2c6-d2b2-a2d5-g4d1')['illegal_move']
)

# Create a red arrow to highlight the illegal move
arrow = chess.svg.Arrow(
    move_obj.from_square,
    move_obj.to_square,
    color="#FF0000"
)

# Render the board position just before the illegal move
# Highlight the origin and destination squares
svg_board = chess.svg.board(
    board=analyze_illegal_sequence('f2h8-f2c6-d2b2-a2d5-g4d1')['board_before_error'],
    arrows=[arrow],
    squares=[move_obj.from_square, move_obj.to_square],
    size=400
)

svg_board
```

![image.png](Chess%20learning/image%201.png)

A pawn cannot legally move that far across the board. 

This method for generating invalid sequences is too naive, as it is highly likely that even the very first move is illegal.

In fact:

```python
# List to store the probability that a randomly generated sequence is legal
prob_valid = []

# Test sequence lengths from 1 to 20 moves
for n_moves in range(1, 21):
    
    # Counter for valid sequences
    valid_count = 0
    
    # Generate 50,000 random sequences of length n_moves
    for _ in range(50000):
        
        # Generate a fully random sequence (no legality constraints)
        # and check whether it is actually legal
        experiment = push_sequence_of_moves(generate_random_sequence(n_moves))
        
        # If the function does not return None, the sequence is legal
        if experiment is not None:
            valid_count += 1
    
    # Estimate the probability of legality for sequences of length n_moves
    prob_valid.append(valid_count / 50000)

# Print the estimated probabilities
print(prob_valid)
```

`[0.00474, 2e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]`

The results clearly show that the probability of generating a legal sequence at random drops to nearly zero after just one move. Note that this exactly what is expected since at the beginning we have 20 legal possible moves out of 4032.

In fact, fewer than 5 out of 1000 randomly generated first moves are legal. From the second move onward, the probability of maintaining legality becomes essentially negligible.

Therefore, instead of relying on purely random sequences, we introduce a more structured strategy: starting from legal sequences and corrupting a single move at a random position. This approach creates more challenging examples, encouraging the neural network to learn deeper structural patterns rather than simply detecting trivial illegality.

```python
def corrupt_valid_sequence(valid_sequence):

    # Generate all possible board squares (e.g., "a1", "h8")
    letters = ['a','b','c','d','e','f','g','h']
    numbers = list(range(1,9))
    positions = list(
        map(lambda el: el[0] + str(el[1]),
            list(itertools.product(letters, numbers)))
    )
    
    # Split the valid sequence into individual moves
    moves = valid_sequence.split('-')
    
    # Randomly choose a position in the sequence to corrupt
    k = random.randint(1, len(moves))

    # Reconstruct the board up to move k-1
    trimmed_sequence = '-'.join(moves[0:k-1])
    new_board_k = push_sequence_of_moves(trimmed_sequence)

    # Extract all legal moves in the current board position
    legal_moves = [m.uci() for m in new_board_k.legal_moves]

    # Generate a random move that is NOT legal in the current position
    # Start from a legal move and keep sampling until an illegal one is found
    random_move = legal_moves[0]
    while random_move in legal_moves:
        random_move = generate_random_sequence(1)

    # Replace the k-th move with the illegal move
    corrupted_moves = moves.copy()
    corrupted_moves[k-1] = random_move

    # Reconstruct the corrupted sequence
    corrupted_sequence = '-'.join(corrupted_moves)

    return corrupted_sequence
```

Let us consider the legal sequence `'g1f3-c7c6-f3d4-d8a5-d4c6'`. Applying the function defined above, we obtain:

```python
corrupt_valid_sequence('g1f3-c7c6-f3d4-d8a5-d4c6')
'g1f3-c7c6-f3d4-h7d4-d4c6'
```

The move d8a5 has been replaced with h7d4, rendering the sequence illegal. Indeed:

```python
# Convert the illegal move (in UCI format) into a Move object
move_obj = chess.Move.from_uci(
    analyze_illegal_sequence('g1f3-c7c6-f3d4-h7d4-d4c6')['illegal_move']
)

# Create a red arrow to highlight the illegal move
arrow = chess.svg.Arrow(
    move_obj.from_square,
    move_obj.to_square,
    color="#FF0000"
)

# Render the board position just before the illegal move
# Highlight the origin and destination squares
svg_board = chess.svg.board(
    board=analyze_illegal_sequence('g1f3-c7c6-f3d4-h7d4-d4c6')['board_before_error'],
    arrows=[arrow],
    squares=[move_obj.from_square, move_obj.to_square],
    size=400
)

svg_board
```

![image.png](Chess%20learning/image%202.png)

that is clearly illegal.

We now proceed to generate the dataset

```python
# Initialize containers for sequences and corresponding labels
moves = []
validity = []

# Generate 100,000 samples
for _ in range(100000):
    
    # Randomly choose the sequence length between 2 and 20 moves
    n_moves = random.randint(2, 20)
    
    # With approximately 50% probability, generate a legal sequence
    if np.random.uniform(low=-1, high=1, size=1) < 0:
        mv = generate_legal_random_sequence(n_moves)  # Legal sequence
        moves.append(mv)
        validity.append('valid')
    
    # Otherwise, generate a legal sequence and corrupt it
    else:
        mv = generate_legal_random_sequence(n_moves)  # Start from a legal sequence
        moves.append(corrupt_valid_sequence(mv))      # Introduce an illegal move
        validity.append('invalid')

# Create a DataFrame containing sequences and labels
raw_df = pd.DataFrame.from_dict({'string': moves, 'label': validity})

# Display the first rows of the dataset
raw_df.head()
```

![image.png](Chess%20learning/image%203.png)

We now need to preprocess the data in order to train our first model.
The first step consists of building a vocabulary containing all possible moves.

Specifically, we consider all strings formed by concatenating two board squares in UCI format: `<from_square><to_square>`. From an implementation perspective:

```python
# Define board files (columns) and ranks (rows)
letters = ['a','b','c','d','e','f','g','h']
numbers = list(range(1,9))

# Generate all board squares (e.g., "a1", "h8")
squares = list(
    map(lambda el: el[0] + str(el[1]),
        list(itertools.product(letters, numbers)))
)

# Standard moves (no promotion)
normal_moves = [
    from_sq + to_sq
    for from_sq in squares
    for to_sq in squares
    if from_sq != to_sq
]

# White promotions (from rank 7 to rank 8)
white_promotions = []
for file in letters:
    from_sq = file + "7"
    for to_file in letters:
        to_sq = to_file + "8"
        if from_sq != to_sq:
            for piece in ["q","r","b","n"]:  # Promotion pieces
                white_promotions.append(from_sq + to_sq + piece)

# Black promotions (from rank 2 to rank 1)
black_promotions = []
for file in letters:
    from_sq = file + "2"
    for to_file in letters:
        to_sq = to_file + "1"
        if from_sq != to_sq:
            for piece in "qrbn":  # Promotion pieces
                black_promotions.append(from_sq + to_sq + piece)

# Combine all possible move strings
all_moves = normal_moves + white_promotions + black_promotions

# Create token-to-id mapping (start indexing from 1)
token_to_id = {move: i + 1 for i, move in enumerate(all_moves)}

# Vocabulary size (+1 reserved for padding token)
vocab_size = len(token_to_id) + 1

print("Total vocabulary size:", vocab_size)
```

`Total vocabulary size: 4545`

We now encode the vocabulary as integers in order to obtain a numerical representation of the moves.

```python
# Maximum sequence length (fixed input size for the model)
max_len = 20

def encode_and_pad(seq):
    # Split the move sequence into individual tokens
    tokens = seq.split('-')
    
    # Convert each move token into its corresponding integer ID
    ids = [token_to_id[t] for t in tokens]
    
    # If the sequence is shorter than max_len, pad with zeros
    # (0 is reserved as the padding token)
    if len(ids) < max_len:
        ids = ids + [0] * (max_len - len(ids))
    
    # If the sequence is longer than max_len, truncate it
    else:
        ids = ids[:max_len]
        
    return ids
```

We can now construct the feature matrix and label vector for model training.

```python
# Encode each move sequence into a fixed-length integer vector
# and convert the resulting list into a NumPy array
X = np.array(
    raw_df["string"]
    .apply(encode_and_pad)
    .tolist()
)

# Convert textual labels into binary targets:
# "valid"  -> 1
# "invalid" -> 0
y = np.array(
    raw_df["label"]
    .map(lambda x: 1 if x == "valid" else 0)
)
```

As usual, we split the dataset into training and test sets.

```python
from sklearn.model_selection import train_test_split

# Split the dataset into training and test sets
# We also keep the original move sequences (raw strings) for later error analysis
X_train, X_test, y_train, y_test, seq_train, seq_test = train_test_split(
    X,
    y,
    raw_df["string"].values,
    test_size=0.2,      # 20% of the data reserved for testing
    random_state=42,    # Ensures reproducibility
    stratify=y          # Preserve class distribution in both splits
)
```

We can now design a simple baseline model to explore how to approach the problem.

```python
import tensorflow as tf
from tensorflow.keras import layers, models

# Dimension of each embedding vector (latent representation of a move)
embedding_dim = 64  

# Fixed length of input sequences (after padding/truncation)
input_length = 20  

# Define a simple Sequential model
model = models.Sequential()

# Embedding layer:
# Maps each move ID to a dense vector of dimension 'embedding_dim'
model.add(layers.Embedding(input_dim=vocab_size,
                           output_dim=embedding_dim))

# Flatten layer:
# Converts the (sequence_length, embedding_dim) tensor
# into a single long vector for fully connected layers
model.add(layers.Flatten())

# Fully connected layers (MLP)
model.add(layers.Dense(64, activation='relu'))  
model.add(layers.Dense(32, activation='relu'))  
model.add(layers.Dense(16, activation='relu'))  

# Output layer:
# Single neuron with sigmoid activation for binary classification
model.add(layers.Dense(1, activation='sigmoid')) 

# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

We now proceed with training the model:

```python
model.fit(
    X_train,
    y_train,
    epochs=25,
    batch_size=128,
    validation_split=0.2
)
```

To monitor performance on the validation set during training:

```python
pd.DataFrame(model.history.history).plot()
```

![image.png](Chess%20learning/image%204.png)

The validation accuracy stabilizes around 90%, which indicates a reasonably good performance. However, the validation loss continues to increase, providing a clear sign of overfitting.

This suggests that while the model correctly classifies most validation examples, it becomes increasingly overconfident on its mistakes.

Let us now evaluate the performance on the test set.

```python
# Compute predicted probabilities on the test set
y_pred_prob = model.predict(X_test).ravel()

# Convert probabilities into binary class predictions
y_pred_class = (y_pred_prob > 0.5).astype(int)

# Compute test accuracy manually
sum(y_pred_class == y_test) / len(y_test)
```

resulting 89% of accuracy that coherent with the validation accuracy.

While global metrics provide a useful overview, they do not reveal how the model fails.

We therefore analyse the misclassified sequences at the move level, reconstructing the board state to inspect precisely where and why the model makes mistakes.

```python
# Build a diagnostic DataFrame to analyze model predictions
diagnostic_df = pd.DataFrame({
    "sequence": seq_test,            # Original move sequence (UCI string)
    "y_true": y_test,                # Ground truth label
    "y_pred_prob": y_pred_prob,      # Predicted probability (sigmoid output)
    "y_pred_class": y_pred_class     # Binary predicted class
})

# Flag misclassified samples
diagnostic_df["is_error"] = (
    diagnostic_df["y_true"] != diagnostic_df["y_pred_class"]
)
```

We can now explore some descriptive statistics.
Let us begin by investigating whether certain pieces are more frequently associated with misclassifications.

```python
def find_piece(seq):
    test = analyze_illegal_sequence(seq)
    if test is None:
        return 'legal_sequence'
    square = chess.parse_square(test['illegal_move'][0:2])
    piece = test['board_before_error'].piece_at(square)
    if piece:
        return str(piece).lower()
    else:
        return 'empty_square'

# Compute frequency of pieces involved in errors
piece_counts = (
    diagnostic_df[diagnostic_df['is_error']]['sequence']
    .apply(find_piece)
    .value_counts()
    .sort_values()
)

import matplotlib.pyplot as plt

plt.figure()

ax = piece_counts.plot(kind='bar')

plt.xlabel("")
plt.ylabel("Number of Errors")

# Rotate x-axis labels
plt.xticks(rotation=45)

plt.show()
```

![image.png](Chess%20learning/image%205.png)

As shown in the plot above, most misclassifications correspond to illegal sequences that the model incorrectly labels as valid. In particular, a large portion of these errors involve moves originating from empty squares.

This indicates that the model has not fully captured one of the most fundamental constraints of chess: a move must originate from a square that contains a piece.

Among actual pieces, pawns are the most frequently involved in misclassified moves. At first glance, this could simply reflect the fact that many squares are occupied by pawns, especially in the early stages of a game. However, even when aggregating the errors across all other piece types, pawn-related misclassifications remain dominant.

This suggests that pawn dynamics — including their asymmetric movement and capture rules — pose a particular challenge for the model.

We now examine the distribution of the square contents from which misclassified moves originate, stratified by the illegal move index.

The goal is to determine whether certain positions in the sequence are more likely to be associated with specific piece types (or empty squares) when the model makes an error.

```python
# Select only false positives 
# (illegal sequences predicted as legal)
false_positives = diagnostic_df[
    (diagnostic_df["y_true"] == 0) &
    (diagnostic_df["y_pred_class"] == 1)
].copy()

# Extract the index of the illegal move
false_positives["illegal_index"] = false_positives["sequence"].apply(
    lambda x: analyze_illegal_sequence(x)['illegal_move_index']
)

# Extract the piece involved
false_positives["piece"] = false_positives["sequence"].apply(find_piece)

# Contingency table
contingency_table = pd.crosstab(
    false_positives["illegal_index"],
    false_positives["piece"]
)

# Chi-square test of independence
from scipy.stats import chi2_contingency
chi2, p, dof, expected = chi2_contingency(contingency_table)

# Plot stacked distribution
plt.figure(figsize=(10,6))
ax = contingency_table.plot(kind='bar', stacked=True)

plt.xlabel("Illegal Move Index (0-based)")
plt.ylabel("Frequency")
plt.title("Illegal Move Position by Piece Type")

# Force integer x-axis labels
ax.set_xticks(range(len(contingency_table.index)))
ax.set_xticklabels(contingency_table.index.astype(int), rotation=0)

# Move legend outside
ax.legend(title="Piece",
          bbox_to_anchor=(1.05, 1),
          loc='upper left',
          fontsize=9)

# Add p-value annotation inside the plot
ax.text(
    0.02, 0.95,
    f"Chi-square p-value = {p:.4e}",
    transform=ax.transAxes,
    fontsize=8,
    verticalalignment='top',
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
)

plt.show()
```

![image.png](Chess%20learning/image%206.png)

As indicated by the χ² test, and visually supported by the plot, we do not find statistically significant differences in the distribution of piece types across illegal move indices.

This suggests that the model’s errors are not strongly dependent on the position of the illegal move within the sequence. In other words, the model appears to fail in a relatively uniform manner across different phases of the game (from the opening to later moves within the 20-move window).

Out of curiosity, we can also examine whether the model’s errors are more frequent during White’s or Black’s turns.

```python
false_positives["player"] = false_positives["illegal_index"].apply(
    lambda x: "white" if x % 2 == 0 else "black"
)

player_counts = false_positives["player"].value_counts()

plt.figure()
player_counts.plot(kind="bar")

plt.xlabel("Player")
plt.ylabel("Number of Errors")
plt.title("Distribution of Errors by Player")

plt.xticks(rotation=0)
plt.show()
```

![image.png](Chess%20learning/image%207.png)

There is no substantial difference in the frequency of errors between White and Black moves.

However, we can further investigate whether the model has implicitly learned the alternation of turns. To do so, we analyse how often the model incorrectly accepts a move played by the wrong player (i.e., a move that violates the turn order constraint).

```python
def is_wrong_turn(seq):
    # Analyze the first illegal move in the sequence
    result = analyze_illegal_sequence(seq)
    
    board = result["board_before_error"]
    move = result["illegal_move"]
    
    # Extract the origin square of the illegal move
    square = chess.parse_square(move[:2])
    piece = board.piece_at(square)
    
    # If the origin square is empty, this is not a turn-related error
    if piece is None:
        return False
    
    # Return True if the piece color does NOT match the side to move
    return piece.color != board.turn

# Apply the function to false positives
false_positives["wrong_turn"] = false_positives["sequence"].apply(is_wrong_turn)

# Compute the proportion of wrong-turn errors
wrong_turn_rate = false_positives["wrong_turn"].mean()

print("Wrong-turn error rate:", wrong_turn_rate)
```

`Wrong-turn error rate: 0.20470438652256834`

The proportion of wrong-player errors is approximately 20%.

This suggests that the model has, to some extent, internalized the alternation constraint of the game. This is a particularly interesting result, as the model architecture does not explicitly encode sequential structure or turn dynamics. Therefore, the model appears to have learned this rule implicitly from statistical regularities in the data.

Let’s now experiment with a more powerful neural network trained on a larger and richer dataset in order to evaluate whether increased model capacity and data scale lead to measurable performance improvements.

```python
# Initialize containers for sequences and corresponding labels
moves = []
validity = []

# Generate 100,000 samples
for _ in range(300000):
    
    # Randomly choose the sequence length between 2 and 20 moves
    n_moves = random.randint(2, 40)
    
    # With approximately 50% probability, generate a legal sequence
    if np.random.uniform(low=-1, high=1, size=1) < 0:
        mv = generate_legal_random_sequence(n_moves)  # Legal sequence
        moves.append(mv)
        validity.append('valid')
    
    # Otherwise, generate a legal sequence and corrupt it
    else:
        mv = generate_legal_random_sequence(n_moves)  # Start from a legal sequence
        moves.append(corrupt_valid_sequence(mv))      # Introduce an illegal move
        validity.append('invalid')

# Create a DataFrame containing sequences and labels
raw_df = pd.DataFrame.from_dict({'string': moves, 'label': validity})

# Display the first rows of the dataset
raw_df.head()
```

```python
X = np.array(raw_df["string"].apply(encode_and_pad).tolist())
y = np.array(raw_df["label"].map(lambda x: 1 if x == "valid" else 0))

# Split the dataset into training and test sets
# We also keep the original move sequences (raw strings) for later error analysis
X_train, X_test, y_train, y_test, seq_train, seq_test = train_test_split(
    X,
    y,
    raw_df["string"].values,
    test_size=0.2,      # 20% of the data reserved for testing
    random_state=42,    # Ensures reproducibility
    stratify=y          # Preserve class distribution in both splits
)
```

To mitigate overfitting and improve training stability, I introduced dropout to randomly deactivate a fraction of neurons during training, applied L2 regularisation to penalise large weights, and incorporated batch normalisation to stabilise the distribution of activations across mini-batches.

```python
from tensorflow.keras import regularizers, callbacks

embedding_dim = 128  # Size of each move embedding vector

model = models.Sequential()

# Embedding layer:
# Transforms each move ID into a dense vector representation
model.add(layers.Embedding(
    input_dim=vocab_size,      # Total number of possible move tokens
    output_dim=embedding_dim   # Dimension of embedding vectors
))

# Flatten layer:
# Converts the (sequence_length, embedding_dim) matrix
# into a single long feature vector
model.add(layers.Flatten())

# First Dense block
# L2 regularization penalizes large weights to reduce overfitting
model.add(layers.Dense(
    192,
    activation=None,
    kernel_regularizer=regularizers.l2(1e-4)
))
model.add(layers.BatchNormalization())   # Stabilizes activations across mini-batches
model.add(layers.Activation('relu'))     # Non-linearity
model.add(layers.Dropout(0.4))           # Randomly drops 40% of neurons during training

# Second Dense block
model.add(layers.Dense(
    96,
    activation=None,
    kernel_regularizer=regularizers.l2(1e-4)
))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.4))

# Third Dense layer (smaller representation)
model.add(layers.Dense(48, activation='relu'))

# Output layer:
# Single neuron with sigmoid activation for binary classification
model.add(layers.Dense(1, activation='sigmoid'))

# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Early stopping:
# Stops training if validation loss does not improve for 6 epochs
early_stop = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=6,
    restore_best_weights=True
)

# Train the model
history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=128,
    validation_split=0.2,
    callbacks=[early_stop]
)
```

The performance of this model does not exceed that of the previous one; on the contrary, we observe a slight decrease in validation accuracy. However, the training dynamics indicate a reduction in overfitting.

zIn particular, the gap between training and validation performance is smaller, and the validation loss remains more stable throughout training. This suggests that the regularisation techniques have improved generalisation behaviour, even though they did not lead to a measurable increase in overall accuracy.

```python
pd.DataFrame(model.history.history).plot()
```

![Screenshot 2026-02-25 at 11-25-41 chess_project.png](Chess%20learning/Screenshot_2026-02-25_at_11-25-41_chess_project.png)

So far, we have trained a model that aims to classify sequences of moves as valid or invalid, and it performs reasonably well on this task.

However, a more interesting question arises: is the model actually able to recognise the legal moves of a specific piece in a given position?

In principle, we can test this as follows. Given a sequence of moves leading to a certain board configuration, we can append every possible move from that position to the sequence and let the model classify each extended sequence as valid or invalid. If the model has truly internalised the structural rules of the game, it should assign high validity scores only to moves that are legally allowed in that position.

Let us therefore investigate how the model performs in this more demanding task.

```python
def analyze_model_destinations(model, sequence, square, threshold=0.01, plot=True):
    """
    Analyze model-approved destinations for a given piece.

    Green  = predicted valid AND truly legal
    Red    = predicted valid BUT illegal
    Light blue = selected piece square
    """

    board = push_sequence_of_moves(sequence)

    if board is None:
        return None

    from_square = chess.parse_square(square)
    piece = board.piece_at(from_square)

    if piece is None:
        return None

    true_legal_moves = {
        m.to_square for m in board.legal_moves if m.from_square == from_square
    }

    fill = {}
    predicted_valid = 0
    illegal_predicted = 0

    # -------------------------
    # Build candidate moves
    # -------------------------

    candidate_sequences = []
    candidate_squares = []

    for to_square in chess.SQUARES:
        if to_square == from_square:
            continue

        move = chess.Move(from_square, to_square)
        move_uci = move.uci()

        new_sequence = sequence + "-" + move_uci

        candidate_sequences.append(new_sequence)
        candidate_squares.append(to_square)

    # -------------------------
    # Batch encode
    # -------------------------

    encoded_batch = np.array(
        [encode_and_pad(seq) for seq in candidate_sequences]
    )

    # -------------------------
    # Single batch predict
    # -------------------------

    probs = model.predict(encoded_batch, verbose=0).ravel()

    # -------------------------
    # Process results
    # -------------------------

    for prob, to_square in zip(probs, candidate_squares):

        if prob > threshold:
            predicted_valid += 1

            if to_square in true_legal_moves:
                if plot:
                    fill[to_square] = "#90EE90"  # green
            else:
                illegal_predicted += 1
                if plot:
                    fill[to_square] = "#FF7F7F"  # red

    if predicted_valid == 0:
        error_ratio = 0
    else:
        error_ratio = illegal_predicted / predicted_valid

    if plot:
        fill[from_square] = "#ADD8E6"

        svg_board = chess.svg.board(
            board=board,
            fill=fill,
            size=500
        )
        print(sequence)
        display(svg_board)

    return error_ratio
```

Using the function defined above, we can compute the ratio of truly legal moves among those that the model classifies as valid. In other words, we can measure how often the model’s “approved” moves actually comply with the rules of chess.

Let us now collect a sufficiently large number of observations and analyse the results through a series of statistical summaries and visualisations.

```python
def collect_structural_statistics(model, sequences, threshold=0.01):
    
    results = []

    for seq in sequences:
        
        board = push_sequence_of_moves(seq)
        if board is None:
            continue
        
        seq_len = len(seq.split('-'))
        
        for square in chess.SQUARES:
            
            piece = board.piece_at(square)
            if piece is None:
                continue
            
            # Consider only pieces of the side to move
            if piece.color != board.turn:
                continue
            
            ratio = analyze_model_destinations(
                model,
                seq,
                chess.square_name(square),
                threshold=threshold,
                plot=False
            )
            
            if ratio is None:
                continue
            
            results.append({
                "sequence_length": seq_len,
                "piece": str(piece).lower(),
                "error_ratio": ratio,
                "correct_ratio": 1 - ratio
            })
    
    return pd.DataFrame(results)
```

```python
def generate_structural_test_set(n_samples=10, max_moves=40):
    
    sequences = []
    
    for _ in range(n_samples):
        length = random.randint(2, max_moves)
        sequences.append(generate_legal_random_sequence(length))
    
    return sequences
    
new_sequences = generate_structural_test_set(n_samples=200, max_moves=40)

stats_df = collect_structural_statistics(model, new_sequences)
```

```python
# --- Data preparation ---
piece_means = (
    stats_df
    .groupby("piece")["correct_ratio"]
    .mean()
    .sort_values()
)

length_means = (
    stats_df
    .groupby("sequence_length")["correct_ratio"]
    .mean()
)

# --- Multipanel ---
fig, axes = plt.subplots(1, 2, figsize=(14,5))

# Panel 1: Piece
axes[0].bar(piece_means.index, piece_means.values)
axes[0].set_title("Structural Accuracy by Piece")
axes[0].set_ylabel("Mean Correct Ratio")
axes[0].set_ylim(0, 1)

# Panel 2: Length
axes[1].plot(length_means.index, length_means.values)
axes[1].set_title("Structural Accuracy vs Game Length")
axes[1].set_xlabel("Sequence Length")
axes[1].set_ylim(0, 1)

plt.tight_layout()
plt.show()
```

![stat.png](Chess%20learning/stat.png)

As we can observe, the model performs rather poorly on this task. Despite achieving good accuracy in sequence classification, it has not truly learned the underlying rules of the game.

Moreover, the results clearly show that as the game progresses and the position becomes more complex, the number of errors increases significantly.

Below, we present several illustrative examples of “superhero pieces” or “debilitated pieces” from the model’s perspective. The square highlighted in blue indicates the piece under analysis. The squares highlighted in green represent moves that are both predicted as valid by the model and are truly legal. The squares highlighted in red correspond to moves that the model considers valid but are, in fact, illegal.

![Untitled design(2).png](Chess%20learning/Untitled_design(2).png)

This experiment highlights the difference between statistical pattern recognition and true rule internalisation in neural networks.