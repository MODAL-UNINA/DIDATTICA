import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Load CIFAR-10 dataset and filter for 'airplane' and 'automobile' classes
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Filter dataset for 'airplane' (class 0) and 'automobile' (class 1)
train_filter = np.where((y_train.squeeze() == 0) | (y_train.squeeze() == 1))
test_filter = np.where((y_test.squeeze() == 0) | (y_test.squeeze() == 1))

X_train, y_train = X_train[train_filter], y_train[train_filter]
X_test, y_test = X_test[test_filter], y_test[test_filter]

# Normalize pixel values to [0, 1] and reshape images
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Define a simple CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer for binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Prepare data and train the model
model.fit(X_train, (y_train.squeeze() == 1).astype(int), epochs=5, batch_size=64)

# Predict probabilities for the test set
y_proba = model.predict(X_test)

# Calculate ROC curve and AUC
fpr, tpr, thresholds = roc_curve((y_test.squeeze() == 1).astype(int), y_proba)
roc_auc = auc(fpr, tpr)


# Plot ROC curve
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

