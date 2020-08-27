import glob
import skimage.io as io
import skimage.transform as trans
import numpy as np
import pylab as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import skimage.io as io
import numpy as np
import os
import random

def square_image(img, random = None):
    """ Square Image
    Function that takes an image (ndarray),
    gets its maximum dimension,
    creates a black square canvas of max dimension
    and puts the original image into the
    black canvas's center
    If random [0, 2] is specified, the original image is placed
    in the new image depending on the coefficient,
    where 0 - constrained to the left/up anchor,
    2 - constrained to the right/bottom anchor
    """
    size = max(img.shape[0], img.shape[1])
    new_img = np.zeros((size, size),np.float32)
    ax, ay = (size - img.shape[1])//2, (size - img.shape[0])//2

    if random and not ax == 0:
        ax = int(ax * random)
    elif random and not ay == 0:
        ay = int(ay * random)

    new_img[ay:img.shape[0] + ay, ax:ax+img.shape[1]] = img
    return new_img


def reshape_image(img, target_size):
    """ Reshape Image
    Function that takes an image
    and rescales it to target_size
    """
    img = trans.resize(img,target_size)
    img = np.reshape(img,img.shape+(1,))
    img = np.reshape(img,(1,)+img.shape)
    return img

def normalize_mask(mask):
    """ Mask Normalization
    Function that returns normalized mask
    Each pixel is either 0 or 1
    """
    mask[mask > 0.4] = 1
    mask[mask <= 0.4] = 0
    return mask

def show_image(img):
    plt.imshow(img, cmap=plt.cm.gray)
    plt.show()

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def prepare_dataset(
    path_to_data,
    image_folder,
    mask_folder,
    n_samples,
    as_gray = True
):
    """ Prepare Dataset
    Function that takes path to DataSet folder
    which has image and mask folder
    Each image and mask are transformed to square formats:
    reads both image and mask, creates new image and mask;
    generates random spacing coefficient,
    adds original image and paddings to them to make them square,
    then saves new masks and images and delets originals
    """
    path_to_image = os.path.join(path_to_data, image_folder)
    path_to_mask = os.path.join(path_to_data, mask_folder)

    for i in range(1, n_samples + 1):
        try:
            img_name = os.path.join(path_to_image,"%d.jpg"%i)
            mask_name = os.path.join(path_to_mask, "%d.png"%i)

            coefficient = random.uniform(0, 2)

            img = io.imread(fname = img_name, as_gray = as_gray)
            os.remove(img_name)
            # new_img = square_image(img, random = coefficient)
            new_img = (new_img * 255).astype('uint8')
            io.imsave(fname = img_name, arr = new_img)

            mask = io.imread(fname = mask_name,as_gray = as_gray)
            os.remove(mask_name)
            # new_mask = square_image(mask, random = coefficient)
            new_mask = (new_mask * 255).astype('uint8')
            io.imsave(fname = mask_name, arr = new_mask)

            print("Successfully added paddings to image and mask #%d"%i)
        except:
            print("Adding paddings failed at #%d"%i)

    print("All images and masks were resized to SQUARE format")

def train_generator(
    batch_size,
    train_path,
    image_folder,
    mask_folder,
    target_size,
    image_color_mode = 'grayscale',
    mask_color_mode = 'grayscale'
):
    """ Image Data Generator
    Function that generates batches of data (img, mask) for training
    from specified folder. Returns images with specified pixel size
    Does preprocessing (normalization to 0-1)
    """
    # no augmentation, only rescaling
    image_datagen = ImageDataGenerator(rescale=1. / 255)
    mask_datagen = ImageDataGenerator(rescale=1. / 255)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        seed = 1
    )
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        seed = 1
    )
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        mask = normalize_mask(mask)
        yield (img,mask)

def test_generator(
    test_path,
    num_images,
    target_size,
    name = False,
    as_gray = True
):
    """ Image Data Generator
    Function that generates batches od data for testing from specified folder
    Reads images as grey, makes them square, scales them
    Returns images with specified pixel size
    Does preprocessing (normalization to 0-1)
    """
    for i in range(1, num_images + 1):
        if name:
            img = io.imread(os.path.join(test_path,name), as_gray = as_gray)
        else:
            img = io.imread(os.path.join(test_path,"%d.jpeg"%i),as_gray = as_gray)
        # img = square_image(img)
        img = reshape_image(img, target_size)
        yield img

def save_results(
    save_path,
    npyfile,
    name = False,
    num_class = 2
):
    """ Save Results
    Function that takes predictions from U-Net model
    and saves them to specified folder.
    """
    for i,item in enumerate(npyfile):
        img = normalize_mask(item)
        img = (img * 255).astype('uint8')
        if name:
            io.imsave(os.path.join(save_path,"predict"+name), img)
        else:
            io.imsave(os.path.join(save_path,"%d_predict.png"%(i+1)),img)

def is_file(file_name) -> bool:
    """ Is File
    Check if file exists
    Later used to check if user has pretrained models
    """
    return os.path.isfile(file_name)