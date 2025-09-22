import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

X = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
Y = 4 * X + 1  

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
model.compile(optimizer=optimizer, loss='mean_squared_error')

history = model.fit(X, Y, epochs=1000, verbose=0)

Y_pred = model.predict(X, verbose=0)

weights = model.get_weights()
m = weights[0][0][0]
c = weights[1][0]
print(f"Learned equation: y = {m:.2f}x + {c:.2f}")

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.scatter(X, Y, color="blue", label="Training Data")
plt.plot(X, Y_pred, color="red", label=f"Model Prediction (y={m:.2f}x+{c:.2f})")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Linear Regression Fit")
plt.legend()
plt.grid(True)
