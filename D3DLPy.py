#!/usr/bin/env python
# coding: utf-8

# # Day 3

# In[1]:


import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape((60000, 28, 28, 1)).astype('float32') / 255
x_test = x_test.reshape((10000, 28, 28, 1)).astype('float32') / 255
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Function to build the model with Batch Normalization
def build_model(optimizer):
    model = models.Sequential()

    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# List of optimizers to try
optimizers = ['adam', 'sgd', 'rmsprop']

# Train and evaluate the model with each optimizer
for optimizer_name in optimizers:
    print(f"\nTraining with {optimizer_name} optimizer:")
    
    model = build_model(optimizer_name)
    model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2, verbose=2)

    # Evaluate the model on the test set
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f'Test accuracy with {optimizer_name} optimizer: {test_acc}')


# In[3]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from kerastuner.tuners import RandomSearch

# Load dataset (replace this with your own dataset)
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X, y = iris.data, iris.target

# One-hot encode the labels
y = tf.keras.utils.to_categorical(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the input features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the model-building function for Keras Tuner
def build_model(hp):
    model = keras.Sequential()
    
    # Input layer
    model.add(layers.InputLayer(input_shape=(X_train.shape[1],)))
    
    # Hidden layers
    for i in range(hp.Int('num_layers', min_value=1, max_value=5, step=1)):
        model.add(layers.Dense(units=hp.Int('units_' + str(i), min_value=32, max_value=512, step=32),
                               activation='relu'))
        model.add(layers.Dropout(rate=hp.Float('dropout_' + str(i), min_value=0.0, max_value=0.5, step=0.1)))
    
    # Output layer
    model.add(layers.Dense(units=y_train.shape[1], activation='softmax'))
    
    # Compile the model
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# Instantiate the Keras Tuner RandomSearch
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,  # Adjust as needed
    directory='tuner_results',
    project_name='iris_classification')

# Search for the best hyperparameter configuration
tuner.search(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Get the best hyperparameter configuration
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Build the final model with the best hyperparameters
final_model = tuner.hypermodel.build(best_hps)

# Train the final model
final_model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))


# In[ ]:




