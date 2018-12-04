

from PIL import Image

# cat_image_pathname = 'C:/Users/hp/Desktop/AIPROJECTS/Learn-Train-Model-Tensorflow/Image Recognition (CIFAR-10 Project)/Image/cat1.jpg'
# cat_image = Image.open(cat_image_pathname)
#
# cat_image.show()


display_image_pathname = input('Enter image pathname: ')
display_image = Image.open(display_image_pathname)
display_image.show()