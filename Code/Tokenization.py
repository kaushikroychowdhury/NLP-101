import tensorflow_datasets as tfds
import itertools
from tensorflow.keras.preprocessing.text import Tokenizer
imdb_sentence = []
train_data = tfds.as_numpy(tfds.load("imdb_reviews", split='train'))

for item in train_data:
    imdb_sentence.append(str(item['text']))

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(imdb_sentence)

print(dict(itertools.islice(tokenizer.word_index.items(), 20)))          # Shows First 20 items in word index
