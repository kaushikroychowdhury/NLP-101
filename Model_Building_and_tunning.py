import tensorflow_datasets as tfds
import itertools
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import string
import seaborn as sns
sns.set()


train_data = tfds.as_numpy(tfds.load("imdb_reviews", split='train'))
test_data = tfds.as_numpy(tfds.load("imdb_reviews", split='test'))

stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at",
             "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do",
             "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having",
             "he", "hed", "hes", "her", "here", "heres", "hers", "herself", "him", "himself", "his", "how",
             "hows", "i", "id", "ill", "im", "ive", "if", "in", "into", "is", "it", "its", "itself",
             "lets", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought",
             "our", "ours", "ourselves", "out", "over", "own", "same", "she", "shed", "shell", "shes", "should",
             "so", "some", "such", "than", "that", "thats", "the", "their", "theirs", "them", "themselves", "then",
             "there", "theres", "these", "they", "theyd", "theyll", "theyre", "theyve", "this", "those", "through",
             "to", "too", "under", "until", "up", "very", "was", "we", "wed", "well", "were", "weve", "were",
             "what", "whats", "when", "whens", "where", "wheres", "which", "while", "who", "whos", "whom", "why",
             "whys", "with", "would", "you", "youd", "youll", "youre", "youve", "your", "yours", "yourself",
             "yourselves"]

table = str.maketrans('', '', string.punctuation)
vocab_size = 25000
max_length = 10
trunc_type='post'
padding_type='post'

def sentence_preprocess(data):
    sentences = []
    labels = []
    for item in data:
        sentence = str(item['text'].decode('UTF-8').lower())
        sentence = sentence.replace(",", " , ")
        sentence = sentence.replace(".", " . ")
        sentence = sentence.replace("-", " - ")
        sentence = sentence.replace("/", " / ")
        sentence = sentence.replace("'", " ' ")
        soup = BeautifulSoup(sentence)
        sentence = soup.get_text()

        words = sentence.split()
        filtered_sentence = ""
        for word in words:
            if word not in stopwords:
                filtered_sentence = filtered_sentence + word + " "
        sentences.append(filtered_sentence)
        labels.append(int(item['label']))
    return sentences, labels

train, train_label = sentence_preprocess(train_data)
test, test_label =  sentence_preprocess(test_data)

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(train)
train_sequences = tokenizer.texts_to_sequences(train)        # building sequences
test_sequences = tokenizer.texts_to_sequences(test)
# print(tokenizer.word_index)

word_index = tokenizer.word_index
training_padded = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
test_padded = pad_sequences(test_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

training_padded = np.array(training_padded)
train_label = np.array(train_label)
test_padded = np.array(test_padded)
test_label = np.array(test_label)

### Model Architecture :

model1 = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 8),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

### Model Evaluation Graphs
def plot_graph(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()

# Fitting
num_epochs = 10
history1 = model1.fit(training_padded, train_label, epochs=num_epochs, validation_data=(test_padded, test_label), verbose=2)

plot_graph(history1, 'accuracy')
plot_graph(history1, 'loss')