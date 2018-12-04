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

### Building Linear Regression Model

```python
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

# Train

x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

session = tf.Session()
# Set the global initializer for variable nodes
init = tf.global_variables_initializer()
session.run(init)

# Run our linear mode and pass values
print(session.run(linear_model, {x: x_train }))
```
Output:

`[ 0.  -0.5 -1.  -1.5]`

So when we check out values with y_train (The values we expect) actually we are not very far.

The **_loss_** is consider the difference between the output and the y_train.
Our model is minimized by adjusting the W and b values. By adjusting the slope `W` and our `y` intercept.
So basically our aims to closer the output values to the `y_train` values.
We will create our loss model, as `loss variable` below the `linear_model`
```python
loss = tf.reduce_sum(tf.square(linear_model - y))
```
Here we will take the square of each point and take absolute to find the difference.

```python
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
```
Output:
```python
3.5
```
we get a loss of 3.5 that a very resonable since we started form -0.5

> !Alert if we give w=.5 and b=-.5 then we would have a loss of 31.5
This is why a perdition is important




### Building Linear Regression
Previously we worked on building competition graph. In this part we will train our data, optimize the value of 
w and b, and minimizing the loss value. Our loss was `3.5`. We will try to make this close to zero.

We train our model to minimize the loss. We will adjust the loss that will adjust w and b (tensor variables).
After having the suitable values for w and b our loss will be adjusted close to zero possibly. After that we will feed the x values and get the y values. 
We will use tensorflow core functions, Gradient Descent and optimize function.

We create an optimizer.

`optimzer = tf.train.GradientDescentOptimizer(0.01)`

0.01 is the learning rate. It is how the -.5 and .5 will be modified each time. If make this value very low then it will learn 
very slow and consume timing. In case, we set the learning rate very high we won't get the accurate model, It will be adjusting the value with hight values it will be hard ping the correct result.

```python
loss = tf.reduce_sum(tf.square(linear_model - y))
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
```

It will use the learning rate of `0.01` and it will adjust in way to minimize the `loss` function
> **_loss_** is the difference between the current value of  w and b, and the expected values. Simply it is the **difference between the `y`** but it is highly dependent on `w` and `b`.

>**_Remember_** x and y are placeholder, w and b are tensor variable only place that can be changed (adjusted).

We set the number of time we want to train our model that is called epox.

> If we run for a long time it will take time to train the data. and if don't run it for enough time, It won't be able to learn.

```python
# Loop
# Run the train variable
for i in range (1000):
    session.run(train, {x: x_train, y:y_train})
print(session.run([W,b]))
```

All:
```python
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
print(session.run(loss, {x: x_train, y:y_train}))
```

Output:
```python
[array([-0.9999988], dtype=float32), array([0.9999964], dtype=float32)]
```
w is very close to -1 and b close to +1.

Now we check the loss:
```python
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

# Loop
# Run the train variable
for i in range (1000):
    session.run(train, {x: x_train, y:y_train})
new_W, new_b, new_loss = session.run([W, b, loss], {x:x_train, y: y_train})
print("New W: %s"%new_W)
print("New b: %s"%new_b)
print("New loss: %s"%new_loss)
```

Output:
```python
New W: [-0.9999988]
New b: [0.9999964]
New loss: 8.526513e-12
```
`new w` is close to `-1`, `new b` is close `+1` and `new loss` is very close to zero.

Let's run our model with new data (linear_model). w and b are already optimized and will run with x values.
```python
# send an array of x
print(session.run(linear_model, {x: [10, 20, 30, 40]}))
```
Script:
```python
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
for i in range (1000):
    session.run(train, {x: x_train, y: y_train})
new_W, new_b, new_loss = session.run([W, b, loss], {x: x_train, y: y_train})
# send an array of x
print(session.run(linear_model, {x: [10, 20, 30, 40]}))
```
Output:
```python
[ -8.999992 -18.99998  -28.999968 -38.999958]
```
It trained our Linear Regression Model. 

### Import CIFAR Packages
We need to download the cifar from an external source to use it in this project. You can download the CIFAR-10 dataset from https://www.cs.toronto.edu/~kriz/cifar.html. This is provided as open-source.

> The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images 
The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.
> (CIFAR-10 URL: https://www.cs.toronto.edu/~kriz/cifar.html).

Here we have the basic 10 categories such as airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

If we want to make this complex we can work with CIFAR-100 dataset, which support hundred categories. CIFAR-100 consumes much time than CIFAR-10. In this project we want to train fast so will use CIFAR-10.

Scroll down to the download part and download the python version. Download Link : https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
Install Keras by using pip command in terminal or command prompt `pip install Keras`
we need to put the cifar-10 in this location `C:\Program Files\Python36\Lib\site-packages\keras\datasets` and unzip it .

![enter image description here](https://ibb.co/HtBr73q)

Now we will modify the `cifar-10.py` and `cifar.py` with pycharm.
The cifar.py script load the model int the cifar-10 dataset. We are not going to change cifar.py, but we need to modify the cifar-10.py.

 - Line 20: `dirname = 'cifar-10-batches-py'`  add also `dirname = 'C:\Program Files\Python36\Lib\site-packages\keras\datasets\cifar-10-batches-py'`
 the project directory where it is located.

Create the python Script file called `Image-Recognition-Trainer.py`


## Displaying Images with PIL
We are going to import images, and manipulate them. We are using a python image library called PIL, to import images, open images and display them.

We will import the:
`from PIL import Image` Incase you get any error this packages might not be installed. You can install it by going into project interpreter an click on the `+` add button search for the `PIL` library and install it. 

In the 

```python
# Import the library
from PIL import Image  

# Image Location (changing upon)
cat_image_pathname = 'C:/Users/hp/Desktop/AIPROJECTS/Learn-Train-Model-Tensorflow/Image Recognition (CIFAR-10 Project)/Image/cat1.jpg'  

# Image
cat_image = Image.open(cat_image_pathname)  

#Displaying Image  
cat_image.show()
```

This script will display the image. 
In the following script we take the image from the user
```python
from PIL import Image 

# Image Location recieved from user
display_image_pathname = input('Enter image pathname: ')  

# Image
display_image = Image.open(display_image_pathname)  

#Displaying Image  
display_image.show()
```