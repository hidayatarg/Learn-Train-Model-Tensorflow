# Build and train a data model to recognize objects in images

## Install Python 3.6.0
Install python pay attention to check `add-to Path variable` on the installion screen. Select the advance settings and install it for all users. It will install python under the ` C:\program files\python`.

## Install Pycharm 
Pycharm community edition is a good IDE to start the projects. It has good intelisense.
It will be great if you take a small tutorial on how to you pycharm.


## Install Tensorflow 

 - TensorFlow is an open-source machine learning library for research
   and production.  You can visit https://www.tensorflow.org/tutorials/
   for more details. 
 - Tensors are basic units of data and are arrays of
   primitive values.
 - Progams have two sections: building graphs with nodes, and running the graphs.
 - Building:
  	 - Build a graph of nodes 
	 - Specify model type, input parameters, expected output and parameters we want model to optimize
	 - Might also want some training data and some testing data
 - Running:
	 - Run the sesion on our graphs to output any results
	 - Specify how many times we want our model to run
	 - First train with training data then use testing data to asses accuracy 
- One of the simplest models is linear regression. We basically estimate a  line that fits some data and run our mode until the line adjust to be fit the data.
### Constant and Operation nodes
we will explore all the different kinds nodes that we are going to build our linear regression model. In this section we will explain constant and operation nodes. There are the most simple types, They hold some values or they result some operations
```sh 
import tensorflow as tf

#datatype of float 34 will be used to access decimals
const_node_1=tf.constant(1.0, dtype=tf.float32)
const_node_1=tf.constant(2.0)

print(const_node_1)
print(const_node_2)
```
the output will be
```sh
Tensor("Const:0", shape=(), dtype=float32)
Tensor("Const_1:0", shape=(), dtype=float32)
```




to print these values we need to run sessions
```sh 
import tensorflow as tf

const_node_1=tf.constant(1.0, dtype=tf.float32)
const_node_1=tf.constant(2.0)

print(const_node_1)
print(const_node_2)

session=tf.Session()  
print(session.run([const_node_1,const_node_2]))
```
output:
```sh
[1.0, 2.0]
```

We add one more node
```sh

import tensorflow as tf   
const_node_1=tf.constant(1.0, dtype=tf.float32)  
const_node_2=tf.constant(2.0, dtype=tf.float32)  
const_node_3=tf.constant([3.0, 4.0, 5.0], dtype=tf.float32)  
  
session=tf.Session()  
print(session.run(const_node_1))  
print(session.run(const_node_2))  
print(session.run(const_node_3))
```
