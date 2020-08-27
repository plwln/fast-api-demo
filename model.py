import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization, Activation, MaxPooling2D, Dropout, concatenate
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import *

def dice_coef(target, prediction, axis=(1, 2), smooth=0.01):
      """
      Sorenson Dice
      \frac{  2 \times \left | T \right | \cap \left | P \right |}{ \left | T \right | +  \left | P \right |  }
      where T is ground truth mask and P is the prediction mask
      """
      prediction = K.round(prediction)  # Round to 0 or 1

      intersection = tf.reduce_sum(target * prediction, axis=axis)
      union = tf.reduce_sum(target + prediction, axis=axis)
      numerator = tf.constant(2.) * intersection + smooth
      denominator = union + smooth
      coef = numerator / denominator

      return tf.reduce_mean(coef)
def dice_coef_loss(target, prediction, axis=(1, 2), smooth=1.0):
    """
    Sorenson (Soft) Dice loss
    Using -log(Dice) as the loss since it is better behaved.
    Also, the log allows avoidance of the division which
    can help prevent underflow when the numbers are very small.
    """
    intersection = tf.reduce_sum(prediction * target, axis=axis)
    p = tf.reduce_sum(prediction, axis=axis)
    t = tf.reduce_sum(target, axis=axis)
    numerator = tf.reduce_mean(intersection + smooth)
    denominator = tf.reduce_mean(t + p + smooth)
    dice_loss = -tf.math.log(2.*numerator) + tf.math.log(denominator)

    return dice_loss
''' utils.py
File that defines layers for u-net model.
1. input_tensor - Input layer
2. single_conv - one 2D Convolutional layer
3. double_conv - two Sequential 2D Convolutional layers
4. deconv - one 2D Transposed Convolutional layer
5, pooling - one Max Pooling layer followed by Dropout function
6. merge - concatenates two layers
7. callback - returns a ModelCheckpoint, used in main.py for model fitting
'''

# function that defines input layers for given shape
def input_tensor(input_size):
    x = Input(input_size)
    return x

# function that defines one convolutional layer with certain number of filters
def single_conv(input_tensor, n_filters, kernel_size):
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size), activation = 'sigmoid')(input_tensor)
    return x

# function that defines two sequential 2D convolutional layers with certain number of filters
def double_conv(input_tensor, n_filters, kernel_size = 3):
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size), padding = 'same', kernel_initializer = 'he_normal')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size), padding = 'same', kernel_initializer = 'he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

# function that defines 2D transposed convolutional (Deconvolutional) layer
def deconv(input_tensor, n_filters, kernel_size = 3, stride = 2):
    x = Conv2DTranspose(filters = n_filters, kernel_size = (kernel_size, kernel_size), strides = (stride, stride), padding = 'same')(input_tensor)
    return x

# function that defines Max Pooling layer with pool size 2 and applies Dropout
def pooling(input_tensor, dropout_rate = 0.1):
    x = MaxPooling2D(pool_size = (2, 2))(input_tensor)
    x = Dropout(rate = dropout_rate)(x)
    return x

# function that merges two layers (Concatenate)
def merge(input1, input2):
    x = concatenate([input1, input2])
    return x

# function to create ModelCheckpoint
def callback(name):
    return ModelCheckpoint(name, monitor='loss',verbose=1, save_best_only=True)

class UNet(Model):
    """ U-Net atchitecture
    Creating a U-Net class that inherits from keras.models.Model
    In initializer, CNN layers are defined using functions from model.utils
    Then parent-initializer is called wuth calculated input and output layers
    Build function is also defined for model compilation and summary
    checkpoint returns a ModelCheckpoint for best model fitting
    """
    def __init__(
        self,
        input_size,
        n_filters,
        pretrained_weights = None
    ):
        # define input layer
        input = input_tensor(input_size)

        # begin with contraction part
        conv1 = double_conv(input, n_filters * 1)
        pool1 = pooling(conv1)

        conv2 = double_conv(pool1, n_filters * 2)
        pool2 = pooling(conv2)

        conv3 = double_conv(pool2, n_filters * 4)
        pool3 = pooling(conv3)

        conv4 = double_conv(pool3, n_filters * 8)
        pool4 = pooling(conv4)

        conv5 = double_conv(pool4, n_filters * 16)

        # expansive path
        up6 = deconv(conv5, n_filters * 8)
        up6 = merge(conv4, up6)
        conv6 = double_conv(up6, n_filters * 8)

        up7 = deconv(conv6, n_filters * 4)
        up7 = merge(conv3, up7)
        conv7 = double_conv(up7, n_filters * 4)

        up8 = deconv(conv7, n_filters * 2)
        up8 = merge(conv2, up8)
        conv8 = double_conv(up8, n_filters * 2)

        up9 = deconv(conv8, n_filters * 1)
        up9 = merge(conv1, up9)
        conv9 = double_conv(up9, n_filters * 1)

        # define output layer
        output = single_conv(conv9, 1, 1)

        # initialize Keras Model with defined above input and output layers
        super(UNet, self).__init__(inputs = input, outputs = output)
        
        # load preatrained weights
        if pretrained_weights:
            self.load_weights(pretrained_weights)

    def build(self):
        self.compile(optimizer=Adam(lr=1e-4), loss=dice_coef_loss, metrics=[dice_coef])
        self.summary()

    def save_model(self, name):
        self.save_weights(name)

    @staticmethod
    def checkpoint(name):
        return callback(name)