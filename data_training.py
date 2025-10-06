import os
import numpy as np
from keras import utils, layers, models

label_names = []
dictionary = {}
is_init = False
c = 0

for file in os.listdir():
    if file.endswith(".npy") and file not in ["labels.npy"]:
        if not is_init:
            X = np.load(file)
            size = X.shape[0]
            y = np.array([file.split('.')[0]] * size).reshape(-1, 1)
            is_init = True
        else:
            X = np.concatenate((X, np.load(file)))
            size = np.load(file).shape[0]
            y = np.concatenate((y, np.array([file.split('.')[0]] * size).reshape(-1, 1)))

        label_names.append(file.split('.')[0])
        dictionary[file.split('.')[0]] = c
        c += 1

# Convert labels to numeric
for i in range(y.shape[0]):
    y[i, 0] = dictionary[y[i, 0]]
y = np.array(y, dtype="int32")

# One-hot encode labels
y = utils.to_categorical(y)

# Shuffle data
idx = np.arange(X.shape[0])
np.random.shuffle(idx)
X, y = X[idx], y[idx]

# Build model
ip = layers.Input(shape=(X.shape[1],))
m = layers.Dense(256, activation="relu")(ip)
m = layers.Dense(128, activation="relu")(m)
op = layers.Dense(y.shape[1], activation="softmax")(m)

model = models.Model(inputs=ip, outputs=op)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X, y, epochs=50, batch_size=16)
model.save("posture_model.h5")
np.save("labels.npy", np.array(label_names))

print("\nModel training completed and saved as posture_model.h5")
