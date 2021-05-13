import tensorflow as tf
import csv
import numpy as np
import matplotlib.pyplot as plt
import mpu.ml
import re
import statistics
from tqdm import tqdm
from sklearn.metrics import f1_score

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
    
def ReadFile(input_csv_file):
    # Reads input_csv_file and returns four dictionaries tweet_id2text, tweet_id2issue, tweet_id2author_label, tweet_id2label
    tweet_id2text = {}
    tweet_id2issue = {}
    tweet_id2author_label = {}
    tweet_id2label = {}
    f = open(input_csv_file, "r", encoding='utf-8', newline='')
    csv_reader = csv.reader(f)
    row_count=-1
    for row in csv_reader:
        row_count+=1
        if row_count==0:
            continue

        tweet_id = int(row[0])
        issue = str(row[1])
        text = str(row[2])
        author_label = str(row[3])
        label = row[4]
        tweet_id2text[tweet_id] = text
        tweet_id2issue[tweet_id] = issue
        tweet_id2author_label[tweet_id] = author_label
        tweet_id2label[tweet_id] = label
    return tweet_id2text, tweet_id2issue, tweet_id2author_label, tweet_id2label

def SaveFile(tweet_id2text, tweet_id2issue, tweet_id2author_label, tweet_id2label, output_csv_file, testid_list, test_label):

    with open(output_csv_file, mode='w', encoding='utf-8', newline='') as out_csv:
        writer = csv.writer(out_csv, delimiter=',', quoting=csv.QUOTE_NONE)
        writer.writerow(["tweet_id", "issue", "text", "author", "label"])
        i = 0
        for tid in testid_list:
            writer.writerow([tid, tweet_id2issue[tid], tweet_id2text[tid], tweet_id2author_label[tid], test_label[i]])
            i+=1

def plot_graphs(history, string):
  plt.plot(h.history[string])
  plt.plot(h.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()

train_sent, train_label = generate_set("train.csv")
test_sent, test_label = generate_set("test.csv")
test_sent_new, test_label_new = generate_set("train.csv")
## Pre-process data ##

# Balance the dataset
counter = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
blacklist = []
new_train = []
new_label = []
i = 0
for label in train_label:
    counter[int(label)] += 1
    if counter[int(label)] > 50:
        blacklist.append(i)
    i+=1
for k in range(len(train_sent)):
    if k not in blacklist:
        new_train.append(train_sent[k])
        new_label.append(train_label[k])
train_sent = new_train
train_label = new_label

# Remove special characters and convert to lower case
train_sent = [re.sub('[^A-Za-z0-9 ]+', '', sent).lower() for sent in train_sent]

# Tokenize data
tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token="<OOV>", num_words=10000)
tokenizer.fit_on_texts(train_sent)
seq_train = tokenizer.texts_to_sequences(train_sent)
seq_test = tokenizer.texts_to_sequences(test_sent)

# Check the length of majority of the tweets in the train set.
lengths = [len(t.split(' ')) for t in train_sent]

seq_train = tf.keras.preprocessing.sequence.pad_sequences(seq_train, padding="post", maxlen=25, truncating="post") # 26 is the max length of most of the tweets.
seq_train = np.asarray([np.asarray(x) for x in seq_train])
seq_test = tf.keras.preprocessing.sequence.pad_sequences(seq_test, padding="post", maxlen=25, truncating="post") # 26 is the max length of most of the tweets.
seq_test = np.asarray([np.asarray(x) for x in seq_test])

# Change labels to one-hot
train_label_array = np.asarray([ int(x) for x in train_label ])
train_label_array = np.asarray(train_label_array)
train_label_ohot = mpu.ml.indices2one_hot(train_label_array, nb_classes=18)
train_label_ohot = np.asarray([np.asarray(x) for x in train_label_ohot])

# Use baseline model 
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(output_dim=100, input_dim=10000, input_length=25), # Generate Embedding for our vocab
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(18, activation='softmax')
])

model.summary()
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    metrics=['accuracy']
)

# Train the initial model with labelled data
h = model.fit(
    seq_train, train_label_array,
    validation_split=0.1,
    epochs=40,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
    ]
)

# Read and pre process the unlabelled data
train_unlab_sent, _ = generate_set("train_extra.csv")
# Remove special characters and convert to lowercase
train_unlab_sent = [re.sub('[^A-Za-z0-9 ]+', '', sent).lower() for sent in train_unlab_sent]
# Remove URLs
for i in range(len(train_unlab_sent)):
    train_unlab_sent[i] = re.sub(r'http\S+', '', train_unlab_sent[i])

# Self train the model by soft-predicting the labels and choosing certain tweet samples that match the variance criteria described below
earlystop_count = 2
counter_es = 0
max_accuracy = 0
history_mat = []
for i in tqdm(range(1, 20)):
    train_unlab_subset = train_unlab_sent[1000*i : (1000*(i+1))-1]
    seq_train_unlab_subset = tokenizer.texts_to_sequences(train_unlab_subset)
    seq_train_unlab_subset = tf.keras.preprocessing.sequence.pad_sequences(seq_train_unlab_subset, padding="post", maxlen=25, truncating="post") # 26 is the max length of most of the tweets.
    seq_train_unlab_subset = np.asarray([np.asarray(x) for x in seq_train_unlab_subset])
    preds = model.predict(seq_train_unlab_subset)
    selected_train = []
    selected_label = []
    # For every sentence, we consider the ones that have a high prediction confidence. More precisely, we select those predictions that have a difference greater than 20 times that of the variation of the set with the next best prediction.
    for x in range(len(preds)):
        pred_set = set(preds[x])
        pred_variance = statistics.variance([float(x) for x in preds[x]])
        max_val = max(pred_set)
        pred_set.remove(max(pred_set))
        max_val_2 = max(pred_set)
        temp_var = (max_val - max_val_2)**2
        if temp_var > pred_variance*10:
            selected_train.append(train_unlab_subset[x])
            selected_label.append(np.argmax(preds[x]))
    # We re-tokenize the entire training data according to our new set
    train_sent_new = []
    train_sent_new.extend(train_sent)
    train_sent_new.extend(selected_train)
    train_sent = train_sent_new

    train_label_new = []
    train_label_new.extend(train_label)
    train_label_new.extend(selected_label)
    train_label = train_label_new

    tokenizer.fit_on_texts(train_sent_new)
    seq_train_new = tokenizer.texts_to_sequences(train_sent_new)
    seq_train_new = tf.keras.preprocessing.sequence.pad_sequences(seq_train_new, padding="post", maxlen=25, truncating="post") # 30 is the max length of most of the tweets.
    seq_train_new = np.asarray([np.asarray(x) for x in seq_train_new])

    train_label_new = np.array([ int(x) for x in train_label_new ])
    train_label_new = np.array(train_label_new)
    h = model.fit(
        seq_train_new, train_label_new,
        validation_split=0.1,
        epochs=20,
        verbose=0,
        callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
    ]
    )
    print("")
    print(h.history['val_accuracy'][-1])
    print(h.history['val_loss'][-1])
    history_mat.append(h)
    if h.history['accuracy'][-1] > max_accuracy:
        counter_es = 0
        max_accuracy = h.history['accuracy'][-1]
    else:
        counter_es += 1

    if counter_es == earlystop_count:
        break

val_accuracy_combined = []
train_accuracy_combined = []
val_loss_combined = []
train_loss_combined = []
for i in range(len(history_mat)):
    val_accuracy_combined.append(history_mat[i].history['val_accuracy'][-1])
    train_accuracy_combined.append(history_mat[i].history['accuracy'][-1])
    val_loss_combined.append(history_mat[i].history['val_loss'][-1])
    train_loss_combined.append(history_mat[i].history['loss'][-1])

# Evaluate F1-Score
seq_test_eval = tokenizer.texts_to_sequences(test_sent_new)
seq_test_eval = tf.keras.preprocessing.sequence.pad_sequences(seq_test_eval, padding="post", maxlen=25, truncating="post") # 26 is the max length of most of the tweets.
seq_test_eval = np.asarray([np.asarray(x) for x in seq_test_eval])

preds = np.argmax(model.predict(seq_test_eval), axis=-1)
print('Macro-F1-Score:', f1_score([int(x) for x in test_label_new], preds, average='macro'))


# Save predictions to a file
test_tweet_id2text, test_tweet_id2issue, test_tweet_id2author_label, test_tweet_id2label = ReadFile('test.csv')
fin_test_sent, fin_test_label = generate_set("test.csv")
fin_test_sent = [re.sub('[^A-Za-z0-9 ]+', '', sent).lower() for sent in fin_test_sent]
seq_test_fin = tokenizer.texts_to_sequences(fin_test_sent)
seq_test_fin = tf.keras.preprocessing.sequence.pad_sequences(seq_test_fin, padding="post", maxlen=25, truncating="post") # 26 is the max length of most of the tweets.
seq_test_fin = np.asarray([np.asarray(x) for x in seq_test_fin])
preds_fin = np.argmax(model.predict(seq_test_fin), axis=-1)
testid_list = list(test_tweet_id2text.keys())
SaveFile(test_tweet_id2text, test_tweet_id2issue, test_tweet_id2author_label, test_tweet_id2label, 'test_proj.csv', testid_list, test_label=preds_fin)