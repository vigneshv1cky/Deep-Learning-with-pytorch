import torch
import torch.nn as nn

###############################
# 1. One-hot encoding characters
###############################
text = "hello"

# Create a sorted set of unique characters
chars = sorted(list(set(text)))
char_to_idx = {char: idx for idx, char in enumerate(chars)}
idx_to_char = {idx: char for char, idx in char_to_idx.items()}

vocab_size_chars = len(chars)
# Build one-hot encoded vectors using an identity matrix
one_hot_chars = []
for char in text:
    print(char)
    idx = char_to_idx[char]
    one_hot = torch.eye(vocab_size_chars)[idx]
    one_hot_chars.append(one_hot)

one_hot_chars = torch.stack(one_hot_chars)
print("One-hot encoding for characters:")
print(one_hot_chars)

####################################
# 2. One-hot encoding whole words
####################################
sentence = "this is a test this is only a test"
words = sentence.split()  # tokenize the sentence by whitespace

# Create a sorted set of unique words
unique_words = sorted(list(set(words)))
word_to_idx = {word: idx for idx, word in enumerate(unique_words)}
idx_to_word = {idx: word for word, idx in word_to_idx.items()}

vocab_size_words = len(unique_words)
one_hot_words = []
for word in words:
    idx = word_to_idx[word]
    one_hot = torch.eye(vocab_size_words)[idx]
    one_hot_words.append(one_hot)

one_hot_words = torch.stack(one_hot_words)
print("\nOne-hot encoding for words:")
print(one_hot_words)

###############################
# 3. Text embeddings using nn.Embedding
###############################

# Define embedding dimension (e.g., 8)
embedding_dim = 8
embedding_layer = nn.Embedding(vocab_size_words, embedding_dim)

# Convert the list of words in the sentence to their corresponding indices
word_indices = [word_to_idx[word] for word in words]
word_indices_tensor = torch.tensor(word_indices)

# Generate dense embeddings
embedded_words = embedding_layer(word_indices_tensor)
print("\nText embeddings for words:")
print(embedded_words)
