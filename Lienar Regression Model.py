import tensorflow as tf

# y = Wx + b

# x will be place holder
# m and b will change by model
# W some weight for m

# Create some X values
# Create some Y values

# x = [1, 2, 3,4]
# y = [0, -1, -2, -3]

# these points are smaller not far from 1
W = tf.Variable([-.5], dtype=tf.float32)
b = tf.Variable([.5], dtype=tf.float32)

# We see the number how accurate
# If our guess is far from the real answer the data will take time to train
# If our guess is near to the real data it will take less time to train

x = tf.placeholder(dtype=tf.float32)
y = tf.placeholder(dtype=tf.float32)

linear_model = W * x + b

loss = tf.reduce_sum(tf.square(linear_model - y))

# Train

x_train = [1, 2, 3, 4]
#         [ 0.  -0.5 -1.  -1.5] - The values we received
y_train = [0, -1, -2, -3]


session = tf.Session()
# Set the global initializer for variable nodes
init = tf.global_variables_initializer()
session.run(init)

# Run our linear mode and pass values
# print(session.run(linear_model, {x: x_train}))
print(session.run(loss, {x: x_train, y:y_train}))