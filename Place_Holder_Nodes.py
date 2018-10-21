import tensorflow as tf

# data type to the float 32
placeholer_1 = tf.placeholder(dtype=tf.float32)
placeholer_2 = tf.placeholder(dtype=tf.float32)

multiply_node_1 = placeholer_1 * 3
multiply_node_2 = placeholer_1 * placeholer_2

session = tf.Session()

# We need to provide values for both place holder inorder to multiply them
# If we use single node we do not need to pass the other node a value

print(session.run(multiply_node_1, {placeholer_1: [1.0, 2.0]}))

# we provide a tensor value to the place holder two
print(session.run(multiply_node_2, {placeholer_1: 4.0, placeholer_2: [2.0, 5.0]}))