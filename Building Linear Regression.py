import tensorflow as tf

# y = Wx + b

# x will be place holder
# m and b will change by model
# W some weight for m


W = tf.Variable([-.5], dtype=tf.float32)
b = tf.Variable([.5], dtype=tf.float32)


x = tf.placeholder(dtype=tf.float32)
y = tf.placeholder(dtype=tf.float32)


linear_model = W * x + b

loss = tf.reduce_sum(tf.square(linear_model - y))
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

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
# print(session.run(loss, {x: x_train, y:y_train}))

# Loop
# Run the train variable
for i in range (1000):
    session.run(train, {x: x_train, y: y_train})
new_W, new_b, new_loss = session.run([W, b, loss], {x: x_train, y: y_train})
# print("New W: %s"%new_W)
# print("New b: %s"%new_b)
# print("New loss: %s"%new_loss)

# send an array of x
print(session.run(linear_model, {x: [10, 20, 30, 40]}))