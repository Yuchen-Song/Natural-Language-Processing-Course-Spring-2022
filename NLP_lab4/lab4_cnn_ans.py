from keras.layers import Input, Dense, Embedding, Conv1D, GlobalMaxPooling1D
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.models import Model
from data_helper import load_data
from keras.initializers import Constant

import os
os.environ['CUDA_VISIBLE_DEVICES']=""

print('Loading data')
X_train, y_train, X_test, y_test, weights= load_data()

sequence_length = X_train.shape[1]
num_classes = y_train.shape[1]
vocabulary_size = weights.shape[0]
embedding_dim = weights.shape[1]

print('Vocab Size',vocabulary_size)
print('Embedding Size',embedding_dim)

# X_train -> (num_train_data, MAX_SEQ_LEN), type: numpy.ndarray
# y_train -> (num_train_data, num_classes), type: numpy.ndarray
# X_test -> (num_test_data, MAX_SEQ_LEN), type: numpy.ndarray
# y_test -> (num_test_data, num_classes), type: numpy.ndarray
# MAX_SEQ_LEN is the fixed length of each sentence, if the length of a sentence
# is less than MAX_SEQ_LEN, then, we will pad this sentence.

# CNN parameters
filter_sizes = [2,3,4]
num_filters = 2  # how many filters we have for each type of filter

# training parameters
drop = 0.5
epochs = 50
batch_size = 8

# Model Architecture
# ----------------------------------------#
print("Creating Model...")
inputs = Input(shape=(sequence_length,), dtype='int32')
# inputs -> [batch_size, sequence_length]

emb_layer = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, embeddings_initializer=Constant(weights), input_length=sequence_length)
# emb_layer.trainable = False

embedding = emb_layer(inputs)
# embedding -> [batch_size, sequence_length,embedding_dim]

drop_embed = Dropout(drop)(embedding) 
# dropout at embedding layer

# Add a convolutional layer with kernel_size = filter_sizes[0], activation function be 'relu', strides = 1
# Hints: using keras.layers.Conv1D, see https://keras.io/layers/convolutional/
conv_0 = Conv1D(num_filters, kernel_size=filter_sizes[0], activation='relu',strides=1)(drop_embed)


# Add a max pooling layer
# Hints: using keras.layers.GlobalMaxPooling1D, see https://keras.io/layers/pooling/
maxpool_0 = GlobalMaxPooling1D()(conv_0)



# Add a convolutional layer with kernel_size = filter_sizes[1], activation function be 'relu', strides = 1
# Hints: using keras.layers.Conv1D, see https://keras.io/layers/convolutional/
conv_1 = Conv1D(num_filters, kernel_size=filter_sizes[1], activation='relu',strides=1)(drop_embed)


# Add a max pooling layer
# Hints: using keras.layers.GlobalMaxPooling1D, see https://keras.io/layers/pooling/
maxpool_1 = GlobalMaxPooling1D()(conv_1)


# Add a convolutional layer with kernel_size = filter_sizes[2], activation function be 'relu', strides = 1
# Hints: using keras.layers.Conv1D, see https://keras.io/layers/convolutional/
conv_2 = Conv1D(num_filters, kernel_size=filter_sizes[2], activation='relu',strides=1)(drop_embed)


# Add a max pooling layer
# Hints: using keras.layers.GlobalMaxPooling1D, see https://keras.io/layers/pooling/
maxpool_2 = GlobalMaxPooling1D()(conv_2)

# Add a concatenate layer which cancatenate the output of all max pooling layers
# Hints: use keras.layers.Concatenate, see https://keras.io/layers/merge/
concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])


			
dropout = Dropout(drop)(concatenated_tensor)

# Add a dropout layer
# Hints: use keras.layers.Dropout, see https://keras.io/layers/core/


# Map to classification space
# Hints: use keras.layers.Dense, see https://keras.io/layers/core/
# Argument hints: units=num_classes, activation = 'softmax'
outputs = Dense(units=num_classes, activation='softmax')(dropout)

# End of Model Architecture
# ----------------------------------------#

# this creates a model that includes
model = Model(inputs=inputs, outputs=outputs)

# optimizer
adam = Adam()

model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

print("Traning Model...")
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test))
