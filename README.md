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
 - Programs have two sections: building graphs with nodes, and running the graphs.
 - Building:
  	 - Build a graph of nodes 
	 - Specify model type, input parameters, expected output and parameters we want model to optimize
	 - Might also want some training data and some testing data
 - Running:
	 - Run the session on our graphs to output any results
	 - Specify how many times we want our model to run
	 - First train with training data then use testing data to asses accuracy 
- One of the simplest models is linear regression. We basically estimate a  line that fits some data and run our mode until the line adjust to be fit the data.
### Constant and Operation nodes
we will explore all the different kinds nodes that we are going to build our linear regression model. In this section we will explain constant and operation nodes. There are the most simple types, They hold some values or they result some operations
```Python 
import tensorflow as tf

#datatype of float 34 will be used to access decimals
const_node_1=tf.constant(1.0, dtype=tf.float32)
const_node_1=tf.constant(2.0)

print(const_node_1)
print(const_node_2)
```
the output will be
```Python
Tensor("Const:0", shape=(), dtype=float32)
Tensor("Const_1:0", shape=(), dtype=float32)
```
to print these values we need to run sessions
```Python
import tensorflow as tf

const_node_1=tf.constant(1.0, dtype=tf.float32)
const_node_1=tf.constant(2.0)

print(const_node_1)
print(const_node_2)

session=tf.Session()  
print(session.run([const_node_1,const_node_2]))
```
output:
```Python
[1.0, 2.0]
```

We add one more node
```Python

import tensorflow as tf   
const_node_1=tf.constant(1.0, dtype=tf.float32)  
const_node_2=tf.constant(2.0, dtype=tf.float32)  
const_node_3=tf.constant([3.0, 4.0, 5.0], dtype=tf.float32)  
  
session=tf.Session()  
print(session.run(const_node_1))  
print(session.run(const_node_2))  
print(session.run(const_node_3))
```

#### Addition of nodes
```python
import tensorflow as tf  
const_node_1=tf.constant(1.0, dtype=tf.float32)  
const_node_2=tf.constant(2.0, dtype=tf.float32)  
const_node_3=tf.constant([3.0, 4.0, 5.0], dtype=tf.float32)  
  
add_node_1=tf.add(const_node_1,const_node_2)  
add_node_2=const_node_1+const_node_2  
  
session=tf.Session()  
print(session.run(add_node_1))  
print(session.run(add_node_2))
```

#### Multiplication of nodes
```python
import tensorflow as tf  
  
const_node_1=tf.constant(1.0, dtype=tf.float32)  
const_node_2=tf.constant(2.0, dtype=tf.float32)  
const_node_3=tf.constant([3.0, 4.0, 5.0], dtype=tf.float32)  
  
add_node_1=tf.add(const_node_1,const_node_2)  
add_node_2=const_node_1+const_node_2  
 
Multi_node_v1= const_node_2*const_node_1    
session=tf.Session()  
print(session.run(add_node_1))  
print(session.run(add_node_2))  
print(session.run(Multi_node_v1))
```
### Place Holder Nodes
Place holder nodes contains nodes with no current values when we create them. We pass some value when running session. You can think of these nodes as taking input. for example our linear regression model
`y=mx+b` when `m` and 	`b` are the variable nodes. `x` is place holder node because we will pass the value to it when we run the program. Similarly while test we may pass a value to `y` which is also a place holder node.
Example:
```python
import tensorflow as tf  
  
# data type to the float 32  
placeholer_1 = tf.placeholder(dtype=tf.float32)  
placeholer_2 = tf.placeholder(dtype=tf.float32)  
  
session = tf.Session()  
  
# Here we should provide pass values to the tensor (array)  
# We need to pass the node first then the value to that node  
# print(session.run(placeholer_1: 5.0)) provide a value or a tensor  
print(session.run(placeholer_1, {placeholer_1: [1.0, 2.0]}))
```
####  Multiplication of place holder nodes
```python 
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
```
Output:
```python
[3. 6.]
[8. 20.]
```
### Variable Nodes
Variable node store initial value but can change. We must call an initializer to assign the value. Where with the constant nodes we can not assign any value once the value is stored.
Variable nodes have tons of extra functionalities. It means there are extra functions we can call on the variable nodes.
```python
import tensorflow as tf  
  
variable_node_1 = tf.Variable([5.0], dtype=tf.float32)    
# For demonstration  
const_node_1 = tf.constant([10.0], dtype=tf.float32)  
  
session = tf.Session()  
print(session.run(const_node_1))
```
Output:
`[10.]`
> ***!Alert:*** If we replace the const node by the variable node we will get errors

We may think the we assigned a tensor to the variable node, but we didn't assigned it. So `variable_node_1` doesn't hold that tensor (value).  In order to solve all those errors we need to create a `Global Initializer`
```python
import tensorflow as tf  
  
variable_node_1 = tf.Variable([5.0], dtype=tf.float32)  
# For demonstration  
const_node_1 = tf.constant([10.0], dtype=tf.float32)  
  
session = tf.Session()  
  
# Create an initializer  
init = tf.global_variables_initializer()  
# Run the initializer  
session.run(init)    
print(session.run(variable_node_1))
```
Output:
`[5.]`
> ***!Alert***: Calling once the `Global Initializer` is enough if you use more than one `variable nodes`.

You can also call `global initializer` inside the `session.run()`.
`session.run(tf.global_variables_initializer())`
You can also multiply it with constant node inside the session.run
```python
import tensorflow as tf  
  
variable_node_1 = tf.Variable([5.0], dtype=tf.float32)    
# For demonstration  
const_node_1 = tf.constant([10.0], dtype=tf.float32)  
  
session = tf.Session()  
  
# Create initializer and Run the initializer  
session.run(tf.global_variables_initializer())  
print(session.run(variable_node_1*const_node_1))
```
  Output:
  `[50.]`
  > tf.Variable is a class and tf.constant is not a class.
##### Assign a value to the variable nodes.
Only running the `variable_node_1.assign([10.0])` will not assign the variable node to the tensor parameter.  We need to create and run the initializer.  
```python
# assign a value or tensor to the variable node
session.run(variable_node_1.assign([10.0]))  
print(session.run(variable_node_1))
```
The script will be 
```python
import tensorflow as tf  
  
variable_node_1 = tf.Variable([5.0], dtype=tf.float32)  
const_node_1 = tf.constant([10.0], dtype=tf.float32)  
  
session = tf.Session()  
init = tf.global_variables_initializer()  
# Create initializer  
session.run(init)  
print(session.run(variable_node_1*const_node_1))  
  
session.run(variable_node_1.assign([10.0]))  
print(session.run(variable_node_1))
```
Output:
```python
[50.]
[10.]
```
## Linear Regression

 - Linear regression is one of the simplest machine learning model
   						`y=mx+b` 
  - It will give the y value based on this function. 
  - This a very good prediction that where the line (through some data points to help) should lie.
  - Program optimizes line by adjusting  `m` and `b` until it minimizes loss
	  -	Loses is the differences between an actual y value and the line itself
	  -	minimal loss corresponds with a line that best fits the data as points are on average closer to actual.
- Training our model:
	- Take x value and correspond y values as inputs.
	- Start with guess for `m` and `b` and measure loss.
	- Run program to adjust `m` and `b` to minimize loss based on inputs.
- Final model will fit a good line through data and will be able to  predict correctly `y` value for given`x` input.
***In summary***, We will build the graph then we will move to the training and finally we will use some test data to asses the accuracy of our model.
