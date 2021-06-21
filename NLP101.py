import tensorflow_datasets as tfds
import itertools
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# imdb_sentence = []
train_data = tfds.as_numpy(tfds.load("imdb_reviews", split='train'))
test_data = tfds.as_numpy(tfds.load("imdb_reviews", split='test'))

# for item in train_data:
#     imdb_sentence.append(str(item['text']))
#
# tokenizer = Tokenizer(num_words=5000)
# tokenizer.fit_on_texts(imdb_sentence)
#
# print(dict(itertools.islice(tokenizer.word_index.items(), 20)))          # Shows First 20 items in word index

# Cleaning text
# 1. Strip-out HTML Tags
# 2. Stripping-out Punctuations
# 3. Remove Stopwords

from bs4 import BeautifulSoup
import string

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
vocab_size = 5000
max_length = 4000
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
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
adam = tf.keras.optimizers.Adam(learning_rate=0.0001)
model1.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

### Model Evaluation Graphs
def plot_graph(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()


num_epochs = 100
history1 = model1.fit(training_padded, train_label, epochs=num_epochs, validation_data=(test_padded, test_label), verbose=2)

plot_graph(history1, 'accuracy')
plot_graph(history1, 'loss')

# Exploring Vocabulary Size

# word_count = tokenizer.word_counts
# from collections import OrderedDict
# newlist = (OrderedDict(sorted(word_count.items(), key=lambda t: t[1], reverse=True)))
# xs=[]
# ys=[]
# curr_x = 1
# for item in newlist:
#  xs.append(curr_x)
#  curr_x=curr_x+1
#  ys.append(newlist[item])
# plt.plot(xs,ys)
# plt.show()
#
# plt.plot(xs,ys)
# plt.axis([300,20000,0,400])
# plt.show()


# Exploring Sentence length
# xs=[]
# ys=[]
# current_item=1
# for item in train:
#  xs.append(current_item)
#  current_item=current_item+1
#  ys.append(len(item))
# newys = sorted(ys)
# plt.plot(xs,newys)
# plt.show()


# vocab Size
# embedding dimension = 4th root of vocab size
# reducing dense neuron 24 to 8
# adding droupout               better for bigger architecture
# l2 regularzation              better for bigger architecture


### Embedding Visualization

# reverse_word_index = dict([(value, key)
# for (key, value) in word_index.items()])

# e = model1.layers[0]
# weights = e.get_weights()[0]
# print(weights.shape)

# import io
# out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
# out_m = io.open('meta.tsv', 'w', encoding='utf-8')
# for word_num in range(1, vocab_size):
#  word = reverse_word_index[word_num]
#  embeddings = weights[word_num]
#  out_m.write(word + "\n")
#  out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
# out_v.close()
# out_m.close()
