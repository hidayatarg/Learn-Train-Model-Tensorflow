import tensorflow as tf

variable_node_1 = tf.Variable([5.0], dtype=tf.float32)
const_node_1 = tf.constant([10.0], dtype=tf.float32)

session = tf.Session()
init = tf.global_variables_initializer()
# Create initializer
session.run(init)
print(session.run(variable_node_1 * const_node_1))

session.run(variable_node_1.assign([10.0]))
print(session.run(variable_node_1))