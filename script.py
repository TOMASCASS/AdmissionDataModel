import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow	import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.metrics import r2_score


# Download the csv file into a DataFrame
df = pd.read_csv("admissions_data.csv")
# print(df.head)

# Sets the features parameters and labels
features = df.iloc[:, 1:8]
labels = df.iloc[:, -1]

# Splits data into training and testing sets
features_train_set, features_test_set, labels_train_set, labels_test_set = train_test_split(features, labels, train_size=0.8, random_state=42)

# Create scaler
scaler = StandardScaler()
# Fit the data
features_train_set_scaled = scaler.fit_transform(features_train_set)
features_test_set_scaled = scaler.transform(features_test_set)

def design_model(feature_data):
    # Create Model
    model = Sequential()
    # Get num of features
    #num_features = feature_data.shape[1]
    # Input layer
    #input = tf.keras.Input(shape=(num_features))
    #model.add(input)
    # Hidden Layers
    hidden_layer = layers.Dense(64, activation='relu')
    model.add(hidden_layer)
    model.add(layers.Dropout(0.5))
    hidden_layer_2 = layers.Dense(32, activation='relu')
    model.add(hidden_layer_2)
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1))

    # Optimize Model
    opt = keras.optimizers.Adam(learning_rate=0.005)
    model.compile(optimizer= opt,
                loss = 'mse',
                metrics = ['mae'])
    return model

# Make Model
model = design_model(features_train_set_scaled)

#print(model.summary)

es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 20)

# Fit the model with 100 epochs and batch size of 8
history = model.fit(features_train_set_scaled, labels_train_set.to_numpy(), epochs=100, batch_size=8, verbose = 1, validation_split=0.2, callbacks=[es])

# Evaluate the model on the test set
val_mse, val_mae = model.evaluate(features_test_set_scaled, labels_test_set.to_numpy())
print("Test Loss:", val_mse)
print("Test Accuracy:", val_mae)

# evauate r-squared score
y_pred = model.predict(features_test_set_scaled)

print(r2_score(labels_test_set,y_pred))

# plot MAE and val_MAE over each epoch
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(history.history['mae'])
ax1.plot(history.history['val_mae'])
ax1.set_title('model mae')
ax1.set_ylabel('MAE')
ax1.set_xlabel('epoch')
ax1.legend(['train', 'validation'], loc='upper left')

# Plot loss and val_loss over each epoch
ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(history.history['loss'])
ax2.plot(history.history['val_loss'])
ax2.set_title('model loss')
ax2.set_ylabel('loss')
ax2.set_xlabel('epoch')
ax2.legend(['train', 'validation'], loc='upper left')

plt.show()