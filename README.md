# NLP-101

Learning and exploring NLP ( Natural Language Processing).
I have applied various text preprocessing methods like TOKENIZATION , SEQUENCING and PADDING.
Also build an initial model and try to tune its hyperparams for better performance.

## Reducing Overfitting in Language Models
1. Adjusting the Learning Rate
2. Exploring Vocabulary size
3. Exploring Sentence length
4. Exploring Embedding Dimension (4th root of Vocabulary Size)
5. Exploring the Model Architechture
    a. Reducing number of Neurons for the Dense Layer (approximatly close to embedding Dimension)
    b. Using Droupout           (Only Works for Complex Architecture)
    c. Using Regularization     (Only Works for Complex Architecture)

For this model, hyperparams are as follows :
```python
learning_rate=0.0001        # For ADAM Optimizer
vocab_size = 5000
max_length = 4000
Embedding_Dimension = 8
```

## Model Architecture
```python
model1 = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 8),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
adam = tf.keras.optimizers.Adam(learning_rate=0.0001)
model1.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
```

### Accuracy
![Accuracy](/Model_Evaluation_Viz/accuracy.png)

### Loss
![Loss](/Model_Evaluation_Viz/loss.png)

## Embedding Visualization using Embedding Projector.
TensorFlow Embedding Projector takes vectors.tsv and metadata.tsv as input.
This code will create vectors and metadata files of our dataset.

```python
reverse_word_index = dict([(value, key)
for (key, value) in word_index.items()])

e = model1.layers[0]
weights = e.get_weights()[0]
print(weights.shape)

import io
out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')
for word_num in range(1, vocab_size):
 word = reverse_word_index[word_num]
 embeddings = weights[word_num]
 out_m.write(word + "\n")
 out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
out_v.close()
out_m.close()
```

![embedding](/Embedding_Viz/Embedding_Projector.png)
