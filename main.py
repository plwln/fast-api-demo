from fastapi import FastAPI
from pydantic import BaseModel
import requests
import urllib.request as urllib
import zipfile
import os
from PIL import Image
import io
from starlette.responses import StreamingResponse
from fastapi.responses import FileResponse
import shutil
import uvicorn
import cv2
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import keras
from keras.models import load_model
from keras.models import Model
import tensorflow as tf
from keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization, Activation, MaxPooling2D, Dropout, concatenate
from keras.callbacks import ModelCheckpoint
import keras.backend as K
from keras.optimizers import *

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

# model = load_model('final_unet_model.hdf5')

def unzip(name, url, folder):    
    if folder=='empty':
        folder = 'shots/'
    else:
        folder = "shots/"+folder
        try:
            os.mkdir(folder)
        except:
            shutil.rmtree(name, ignore_errors=True)
        
    filehandle, _ = urllib.urlretrieve(url)
    zip_file_object = zipfile.ZipFile(filehandle, 'r')
    try:
        os.mkdir(folder+name)
    except:
        shutil.rmtree(name, ignore_errors=True)
    print(folder+name+"/")
    zip_file_object.extractall(folder+name+"/")
    for n in os.listdir(folder+name+"/"):
        image = Image.open(folder+name+"/"+n)
        # image.mode = 'I'
        try:
            # image.point(lambda i: i*(1./256)).convert('L').save('shots/'+name+"/"+n.replace(".tif","")+'.jpeg', "JPEG")
            out = image.convert("RGB")
            out.save(folder+name+"/"+n.replace(".tif","")+'.jpeg', "JPEG", quality=90)
        except:
            continue
    images = []
    for band in ('B3', 'B2', 'B4'):
        images.append(cv2.imread(folder+name+'/'+name+'.'+band+'.jpeg'))
    images[0][:,:,0],images[0][:,:,2]  = images[1][:,:,1], images[2][:,:,2]
    out = image.convert('RGB')
    # out.save('shots/'+name+"/"+name+'.color.jpeg', "JPEG", quality=90)
    cv2.imwrite ( folder+name+"/"+name+'.color.jpeg' , images[0])

class Item(BaseModel):
    urls: dict
    folder: str

class Concatenate(BaseModel):
    folder: str

class imageRequest(BaseModel):
    name: str
    band: str
    type: str
    folder: str

app = FastAPI()
origins = [
    "http://localhost",
    "http://localhost:8082",
    "http://localhost/api/unzip",
    "http://localhost:8082/api/get_image",
    "http://localhost/unzip",
    "http://localhost:8082/get_image",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8082, log_level="info")

@app.post('/api/unzip')
def unzip_page(item: Item):
    for name in item.urls.keys():
        unzip(name, item.urls[name], item.folder)


@app.post("/api/get_image")
async def image_endpoint(images: imageRequest):
    # Returns a cv2 image array from the document vector
    # img = Image.open('./'+images.name+"/image.B2.jpeg")
    folder = 'shots/' if images.folder=='empty' else 'shots/'+images.folder
    path = [n for n in os.listdir(folder+images.name+"/") if images.band in n and images.type in n]
    print(path[0])
    return FileResponse(folder+images.name+"/"+path[0])

@app.post("/api/make_all_color")
def make_all_color(folder):
    print(os.listdir(folder))
    for n in os.listdir(folder):
        print(n)
        images=[]
        for band in ('B3', 'B2', 'B4'):
            images.append(cv2.imread(folder+n+'/'+n+'.'+band+'.jpeg'))
        images[0][:,:,0],images[0][:,:,2]  = images[1][:,:,1], images[2][:,:,2]
        cv2.imwrite( folder+n+"/"+n+'.color.jpeg' , images[0])

@app.get("/api/getCookies")
def getCookies():
    return requests.get("http://91.239.142.111:8888/agat/login?name=teamvolg&password=77lS5r&project=agat").headers['set-cookie']

@app.get("/api/startTask")
def startTask():
    return requests.post("http://91.239.142.111:8888/agat/run/b6aa5e41fdf6e8b502b08447216e735d6").status_code

@app.get("/api/concatenate")
def concatenateImage():
    rows = []
    black_image = np.zeros((32,32,3), np.float32)
    big_img = None
    for i in range(37):
        imgs = None
        for j in range(70):
            print("shots/testObject/testObject/test-"+str(i)+"-"+str(j)+"/test-"+str(i)+"-"+str(j)+".color.jpeg")
            try:
                img = cv2.imread("shots/testObject/testObject/test-"+str(i)+"-"+str(j)+"/test-"+str(i)+"-"+str(j)+".color.jpeg")
                img = cv2.resize(img, (32,32), interpolation=cv2.INTER_AREA)
            except:
                img = black_image
            if imgs is None:
                imgs = img
            else:
                imgs = np.concatenate((imgs, img), axis = 1)
        rows.append(imgs)
    for img in rows:
        if big_img is None:
            big_img = img
        else:
            big_img = np.row_stack((big_img, img))
    cv2.imwrite('out.png', big_img)
