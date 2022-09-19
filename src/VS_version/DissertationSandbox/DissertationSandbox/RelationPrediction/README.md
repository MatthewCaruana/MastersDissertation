# Relation Prediction

## Mohammed's implementation

Identify the relation being queried (used CNNs, RNNs & LR)

## My implementation

third-model-300
	self.iterations = 300
    self.dense_embedding = 16
    self.lstm_units = 16
    self.dense_units = 100
    self.word_count = word_count
    self.batch_size= 128

fourth-model-600
    self.iterations = 600
    self.dense_embedding = 16
    self.lstm_units = 16
    self.dense_units = 100
    self.word_count = word_count
    self.batch_size= 128

fifth-model-20
    self.iterations = 20
    self.dense_embedding = 4
    self.lstm_units = 32
    self.dense_units = 100
    self.word_count = word_count
    self.batch_size= 128

    learning rate = 0.001

sixth-model-200
    self.model.add(layers.Embedding(self.word_count, self.dense_embedding))
    self.model.add(layers.Bidirectional(layers.LSTM(self.dense_embedding)))
    self.model.add(layers.Dense(128, activation="relu"))
    self.model.add(layers.Dropout(self.rnn_dropout))
    self.model.add(layers.Dense(self.label_size, activation="softmax"))

    self.iterations = 200
    self.dense_embedding = 64
    self.lstm_units = 64
    self.dense_units = 100
    self.word_count = word_count
    self.batch_size= 128

    learning rate = 0.0001

seventh-model-500
    self.model.add(layers.Embedding(self.word_count, self.dense_embedding))
    self.model.add(layers.Bidirectional(layers.LSTM(self.dense_embedding)))
    self.model.add(layers.Dense(128, activation="relu"))
    self.model.add(layers.Dropout(self.rnn_dropout))
    self.model.add(layers.Dense(self.label_size, activation="softmax"))

    self.iterations = 500
    self.dense_embedding = 64
    self.lstm_units = 64
    self.dense_units = 100
    self.word_count = word_count
    self.batch_size= 128

    learning rate = 0.0001

eighth_model_400
    self.model.add(layers.Embedding(self.word_count, self.dense_embedding))
    self.model.add(layers.Bidirectional(layers.LSTM(self.dense_embedding)))
    self.model.add(layers.BatchNormalization())
    self.model.add(layers.Dense(128, activation="relu"))
    self.model.add(layers.Dropout(self.rnn_dropout))
    self.model.add(layers.Dense(self.label_size, activation="softmax"))

    self.iterations = 400
    self.dense_embedding = 64
    self.lstm_units = 64
    self.dense_units = 100
    self.word_count = word_count
    self.batch_size= 128

    learning rate = 0.0001
