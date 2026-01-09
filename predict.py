import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


new_model = tf.keras.models.load_model('mobilenet_v2_finetuned_95acc.keras')
print(new_model.summary())

X = np.load('X.npy')
y = np.load('y.npy')

print(X.shape)
print(y.shape)

X = (X * 2.0) - 1.0

y_pred = new_model.predict(X)

y_pred_classes = np.argmax(y_pred, axis=1)

conf_mat = tf.math.confusion_matrix(y, y_pred_classes)

print(conf_mat)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()