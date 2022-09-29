# Entity Detection

This is for the process of identifying whether or not a token within a sentence/question is an entity

## Mohammed's Implementation

To train the model:

```
python EntityDetection.py --entity_detection_mode LSTM --fix_embed
or
python EntityDetection.py --entity_detection_mode LSTM --fix_embed --no_cuda
```

To test the model:

```
python top_retrieval.py --trained_model [path/to/trained_model.pt] --entity_detection_mode LSTM
```

### For CRF:

enter crf folder and run bash auto_run.sh

## My Implementation

### Training 

To train model:

### Predicition



BEST MODEL SO FAR FOR ENTITY DETECTION:
Simple Questions: simple_model_first_200

self.input_size = 100
self.hidden_size = 256
self.layer_numbers = 2
self.rnn_dropout = 0.3
self.label_size = 
np.random.seed(456)
self.batch_size = 500
self.iterations = 
self.dense_embedding = 16 # Dimension of the dense embedding
self.lstm_units = 16
self.dense_units = 100
self.word_count = word_count

def create_model(self, mode):
        self.model = keras.Sequential()
        self.model.add(keras.Input(shape=(300,)))
        if mode == "LSTM":
            self.model.add(layers.Embedding(self.word_count, self.dense_embedding, embeddings_initializer="uniform", input_length=300))
            self.model.add(layers.Bidirectional(layers.LSTM(self.lstm_units, recurrent_dropout=self.rnn_dropout, return_sequences=True)))

            self.model.add(layers.Activation("relu"))
            self.model.add(layers.BatchNormalization(epsilon = 1e-05, momentum=0.1))
            self.model.add(layers.Activation("relu"))
            self.model.add(layers.Dropout(self.rnn_dropout))
            self.model.add(layers.Dense(self.label_size))

        self.model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=0.0001), metrics=["accuracy"])
        self.model.summary()


DOI: doi_model


