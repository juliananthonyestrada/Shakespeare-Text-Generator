import random
import numpy as np
import pandas as pd
import tensorflow as tf

# Download Shakespeare text file
filepath = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

# Open file, read in binary, make everything lowercase for easier prediction -> better prediction
text = open(filepath, 'rb').read().decode(encoding='utf-8').lower()

# Subset the text so that training is faster (these numbers are arbitrary)
text = text[300000:800000]

# Get unique characters in the text
characters = sorted(set(text))

# Create dictionaries for character to index and index to character
char_to_index = dict((c, i) for i, c in enumerate(characters))
index_to_char = dict((i, c) for i, c in enumerate(characters))

# Define sequence length and step size
SEQ_LENGTH = 40
STEP_SIZE = 3

'''
# Prepare the input sequences and their corresponding next characters
sentences = []
next_characters = []

for i in range(0, len(text) - SEQ_LENGTH, STEP_SIZE):
    sentences.append(text[i: i + SEQ_LENGTH])
    next_characters.append(text[i + SEQ_LENGTH])

# Prepare the data arrays
x = np.zeros((len(sentences), SEQ_LENGTH, len(characters)), dtype=np.bool_)
y = np.zeros((len(sentences), len(characters)), dtype=np.bool_)

for i, sentence in enumerate(sentences):
    for t, character in enumerate(sentence):
        x[i, t, char_to_index[character]] = 1
    y[i, char_to_index[next_characters[i]]] = 1


# Build the model
model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(128, input_shape=(SEQ_LENGTH, len(characters))))
model.add(tf.keras.layers.Dense(len(characters), activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.01))

# Train the model
model.fit(x, y, batch_size=256, epochs=4)

# Save the model
model.save('shakespeare_text_generator.keras')
'''

# after we have build and saved the model we can comment out 26-57 and simply load it

model = tf.keras.models.load_model('shakespeare_text_generator.keras')

# sample function is used to select one specific character
def sample(preds, temperature=1.0):
    # create an array of type float using our models predicted characters
    preds = np.asarray(preds).astype('float64')
    # adjust the predictions using temperature -> higher temp = riskier predictions
    preds = np.log(preds) / temperature
    # exponentiate and normalize values so that they may be converted to probabilities
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    # selects the index of one character and returns it
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_text(length, temperature):
    # a random starting index is chosen from the text
    start_index = random.randint(0, len(text) - SEQ_LENGTH)
    generated = ''
    sentence = text[start_index: start_index + SEQ_LENGTH]
    generated += sentence
    # enter loop to generate specified number of characters
    for i in range(length):
        x = np.zeros((1, SEQ_LENGTH, len(characters)))
        for t, character, in enumerate(sentence):
            x[0, t, char_to_index[character]] = 1

        # predict next character
        predictions = model.predict(x, verbose=0)[0]
        # pass the predicted character into sample to choose the next character
        next_index = sample(predictions, temperature)
        next_character = index_to_char[next_index]

        # update the generated sentence
        generated += next_character
        sentence = sentence[1:] + next_character
    
    return generated


print(generate_text(1000, 0.5))