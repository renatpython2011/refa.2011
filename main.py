# ==============================
# AI IMAGE CLASSIFIER (FULL CODE)
# TensorFlow + Keras
# ==============================

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# 1. LOAD DATASET
# ------------------------------
print("Loading dataset...")

fashion_mnist = keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# ------------------------------
# 2. PREPROCESS DATA
# ------------------------------
print("Preprocessing data...")

x_train = x_train / 255.0
x_test = x_test / 255.0

# ------------------------------
# 3. BUILD MODEL (AI)
# ------------------------------
print("Building model...")

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),

    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),

    keras.layers.Dense(64, activation='relu'),

    keras.layers.Dense(10, activation='softmax')
])

# ------------------------------
# 4. COMPILE MODEL
# ------------------------------
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ------------------------------
# 5. TRAIN AI
# ------------------------------
print("\nTraining AI...\n")

model.fit(x_train, y_train, epochs=5)

# ------------------------------
# 6. EVALUATE AI
# ------------------------------
print("\nEvaluating AI...\n")

test_loss, test_acc = model.evaluate(x_test, y_test)

print(f"\nTest Accuracy: {test_acc:.4f}")

# ------------------------------
# 7. MAKE PREDICTIONS
# ------------------------------
print("\nMaking predictions...\n")

predictions = model.predict(x_test)

# ------------------------------
# 8. SHOW RESULTS
# ------------------------------
def show_prediction(index):
    plt.imshow(x_test[index], cmap=plt.cm.binary)
    plt.title(f"Predicted: {class_names[np.argmax(predictions[index])]}\n"
              f"Actual: {class_names[y_test[index]]}")
    plt.axis('off')
    plt.show()

# Show a few predictions
for i in range(3):
    show_prediction(i)

# ------------------------------
# 9. SAVE MODEL
# ------------------------------
model.save("ai_model.h5")
print("\nModel saved as ai_model.h5")

# ------------------------------
# 10. LOAD MODEL (OPTIONAL TEST)
# ------------------------------
loaded_model = keras.models.load_model("ai_model.h5")

sample = x_test[0].reshape(1, 28, 28)
prediction = loaded_model.predict(sample)

print("\nLoaded model prediction:",
      class_names[np.argmax(prediction)])