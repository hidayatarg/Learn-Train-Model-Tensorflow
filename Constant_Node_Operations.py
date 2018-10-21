import tensorflow as tf

const_node_1 = tf.constant(1.0, dtype=tf.float32)
const_node_2 = tf.constant(2.0, dtype=tf.float32)
const_node_3 = tf.constant([3.0, 4.0, 5.0], dtype=tf.float32)

add_node_1 = tf.add(const_node_1, const_node_2)
add_node_2 = const_node_1 + const_node_2

Multi_node_v1 = const_node_2 * const_node_1
session = tf.Session()
print(session.run(add_node_1))
print(session.run(add_node_2))
print(session.run(Multi_node_v1))