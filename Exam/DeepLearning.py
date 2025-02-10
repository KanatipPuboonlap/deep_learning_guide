

import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from sklearn.datasets import make_classification

# กำหนด seed ให้คงที่สำหรับ reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)


X, y = make_classification(n_samples=600, n_features=20, n_informative=20, n_redundant=0,
                           n_classes=10, n_clusters_per_class=1, class_sep=3.0,
                           flip_y=0, random_state=42)

# แบ่งข้อมูลเป็น training และ validation
X_train_s, y_train_s = X[:500], y[:500]
X_val_s, y_val_s = X[500:], y[500:]

# ปรับปรุงโมเดลด้านล่างให้มีค่า Accuracy มากขึ้นกว่า 80%:
improved_model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation='relu', input_shape=(20,)),
    tf.keras.layers.Dense(1, activation='relu'),
    tf.keras.layers.Dense(1, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

improved_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])

# ตั้งค่า callbacks สำหรับการฝึกโมเดล
callbacks_improved = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
]

# ฝึกโมเดลกับข้อมูล 
history_improved = improved_model.fit(X_train_s, y_train_s, epochs=1, batch_size=32,
                                      validation_data=(X_val_s, y_val_s),
                                      callbacks=callbacks_improved,
                                      verbose=1)

# ประเมินผลโมเดลที่ปรับปรุงแล้ว
y_pred_improved = np.argmax(improved_model.predict(X_val_s), axis=1)
print("\n--- Improved Model Classification Report ---")
print(classification_report(y_val_s, y_pred_improved))

# แสดง Confusion Matrix สำหรับโมเดลที่ปรับปรุงแล้ว
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_val_s, y_pred_improved), annot=True, cmap='Greens', fmt='d',
            xticklabels=range(10), yticklabels=range(10))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Improved Model Confusion Matrix")
plt.show()
