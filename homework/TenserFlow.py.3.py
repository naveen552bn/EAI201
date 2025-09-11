import tensorflow as tf
import numpy as np

X = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
Y = 4 * X + 1  

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
model.compile(optimizer=optimizer, loss='mean_squared_error')

model.fit(X, Y, epochs=1000, verbose=0)
x_input = float(input("Enter the value of x: "))

y_pred = model.predict(np.array([x_input]), verbose=0)
print(f"\nPredicted y for x = {x_input} is: {y_pred[0][0]:.2f}")

weights = model.get_weights()
m = weights[0][0][0]
c = weights[1][0]
print(f"Learned equation: y = {m:.2f}x + {c:.2f}")


"""
Output:
     Enter the value of x: 4
     Predicted y for x = 4.0 is: 17.00
     Learned equation: y = 4.00x + 1.00
"""
