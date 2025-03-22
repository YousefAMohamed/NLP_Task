import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, GRU, Dense, Dropout

# Load IMDB Dataset
vocab_size = 10000
max_length = 200
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

# Padding sequences (changed padding to 'pre')
x_train = pad_sequences(x_train, maxlen=max_length, padding='pre', truncating='pre')
x_test = pad_sequences(x_test, maxlen=max_length, padding='pre', truncating='pre')

# Model Definition
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=64),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.3),
    GRU(32),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

# Compile the Model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the Model (reduced epochs to 3)
model.fit(x_train, y_train, epochs=3, batch_size=64, validation_data=(x_test, y_test))

# Evaluate the Model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

# Sentiment Prediction
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

def encode_review(text):
    words = text.lower().split()
    encoded = [word_index.get(word, 2) for word in words]  # Use 2 for unknown words
    return pad_sequences([encoded], maxlen=max_length, padding='pre', truncating='pre')

# Test with a sample review (different review text)
review = "The movie was terrible, I regret watching it."
encoded_review = encode_review(review)

prediction = model.predict(encoded_review)[0][0]
sentiment = "Positive" if prediction > 0.5 else "Negative"

print(f"Review: {review}")
print(f"Predicted Sentiment: {sentiment} (Confidence: {prediction:.4f})")