import numpy as np 
from nltk.tokenize import RegexpTokenizer
from keras.models import Sequential, load_model
from keras.layers import LSTM
from keras.layers.core import Dense, Activation
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import pickle
import heapq

path = 'path'
text = open(path, encoding='utf8').read().lower()

tokenizer = RegexpTokenizer(r'\w+')
word = tokenizer.tokenize(text)

unique_word = np.unique(word)
unique_word_index = dict((c, i) for i, c in enumerate(unique_word))

WORD_LENGTH = 5
prev_word = []
next_word = []
for i in range(len(word) - WORD_LENGTH):
    prev_word.append(word[i : i + WORD_LENGTH])
    next_word.append(word[i + WORD_LENGTH])

X = np.zeros((len(prev_word), WORD_LENGTH, len(unique_word)), dtype = bool)
Y = np.zeros((len(next_word), len(unique_word)), dtype = bool)

for i, each_word in enumerate(prev_word):
    for j, each_word in enumerate(each_word):
        X[i, j, unique_word_index[each_word]] = 1
    Y[i, unique_word_index[next_word[i]]] = 1

model = Sequential()
model.add(LSTM(128, input_shape = (WORD_LENGTH, len(unique_word))))
model.add(Dense(len(unique_word)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr = 0.01)
model.compile(loss = 'categorical_crossentropy', optimizer = optimizer , metrics=['accuracy'])
history = model.fit(X, Y, validation_split = 0.05, batch_size = 128, shuffle = True).history

model.save('keras_next_word_model.h5')
pickle.dump(history, open("history.p", "wb"))

model = load_model('keras_next_word_model.h5')
history = pickle.load(open("history.p", "rb"))

def prepare_input(text):
    x = np.zeros((1, WORD_LENGTH, len(unique_word)))
    for a, words in enumerate(text.split()):
        print(words)
        x[0, a, unique_word_index[words]] = 1
    return x

def sample(preds, top_n = 1):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)

    return heapq.nlargest(top_n, range(len(preds)), preds.take)

def predict_completion(text, n = 1):
    if text == '':
        return("0")
    x = prepare_input(text)
    preds = model.predict(x, verbose = 0)[0]
    next_indices = sample(preds, n)
    return [unique_word[idx] for idx in next_indices]

Input =  input("Type here..: ")
print("Correct sentence: ", Input)
seq = " ".join(tokenizer.tokenize(Input.lower())[0:])
print("Sequence: ",seq)
print("Next possible words: ", predict_completion(seq, 5))
