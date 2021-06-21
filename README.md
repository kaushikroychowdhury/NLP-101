# NLP-101

Learning and exploring NLP ( Natural Language Processing).
I have applied various text preprocessing methods like TOKENIZATION , SEQUENCING and PADDING.
Also build an initial model and try to tune its hyperparams for better performance.

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
