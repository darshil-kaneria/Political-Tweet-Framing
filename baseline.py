import tensorflow as tf
import csv
import numpy as np
import matplotlib.pyplot as plt
import mpu.ml
import re

# Read train and test data
def generate_set(input_file):
    dataset = []
    with open(input_file, newline='', encoding="utf-8") as csvfile:
        data = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in data:
            dataset.append(row)
    return [item[2] for item in dataset[1:]], [item[4] for item in dataset[1:]]
    
def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])

def plot_graphs(history, string):
  plt.plot(h.history[string])
  plt.plot(h.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()

train_sent, train_label = generate_set("train.csv")
test_sent, test_label = generate_set("test.csv")

## Pre-process data ##

# Remove special characters and convert to lower case
train_sent = [re.sub('[^A-Za-z0-9 ]+', '', sent).lower() for sent in train_sent]

# Tokenize data
tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token="<OOV>", num_words=4000)
tokenizer.fit_on_texts(train_sent)
seq_train = tokenizer.texts_to_sequences(train_sent)
seq_test = tokenizer.texts_to_sequences(test_sent)

# Check the length of majority of the tweets in the train set.
lengths = [len(t.split(' ')) for t in train_sent]
plt.hist(lengths, bins=len(set(lengths)))
plt.show()

seq_train = tf.keras.preprocessing.sequence.pad_sequences(seq_train, padding="post", maxlen=30, truncating="post") # 26 is the max length of most of the tweets.
seq_train = np.asarray([np.asarray(x) for x in seq_train])
seq_test = tf.keras.preprocessing.sequence.pad_sequences(seq_test, padding="post", maxlen=30, truncating="post") # 26 is the max length of most of the tweets.
seq_test = np.asarray([np.asarray(x) for x in seq_test])

# Change labels to one-hot
train_label = np.array([ int(x) for x in train_label ])
train_label = np.array(train_label)
train_label_ohot = mpu.ml.indices2one_hot(train_label, nb_classes=18)
train_label_ohot = np.asarray([np.asarray(x) for x in train_label_ohot])

# Create baseline model 
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(output_dim=100, input_dim=4000, input_length=30), # Generate Embedding for our vocab
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(18, activation='softmax')
])

model.summary()
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Train the model
h = model.fit(
    seq_train, train_label,
    validation_split=0.1,
    epochs=40,
)

# Evaluate results
plot_graphs(h, 'accuracy')
plot_graphs(h, 'loss')


