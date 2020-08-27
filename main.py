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
import tensorflow.keras
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
# import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization, Activation, MaxPooling2D, Dropout, concatenate
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import *
from model import UNet
from data import *
import  subprocess



img_height = 256
img_width = 256
img_size = (img_height, img_width)
model_name = 'final_unet_model.hdf5'
model_weights_name = 'final_unet_weight_model.hdf5'
model = UNet(
    input_size = (img_width,img_height,1),
    n_filters = 64,
    pretrained_weights = model_weights_name
)

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

class ConcatenateType(BaseModel):
    type: str

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
    model.build()
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

@app.post("/api/concatenate")
def concatenateImage(typeRequest:ConcatenateType):
    
    rows = []
    black_image = np.zeros((32,32,3), np.float32)
    big_img = None
    for i in range(37):
        imgs = None
        for j in range(70):
            path = "shots/testObject/testObject/test-"+str(i)+"-"+str(j)+"/"
            if typeRequest.type == "predicts":
                path = "predicts/predict"
            # print(path+"test-"+str(i)+"-"+str(j)+".color.jpeg")
            try:
                img = cv2.imread(path+"test-"+str(i)+"-"+str(j)+".color.jpeg")
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

@app.post("/api/predict")
def predict(folderRequest: Concatenate):
     for i in range(37):
        for j in range(70):
            try:
                name =  "test-"+str(i)+"-"+str(j)+".color.jpeg"
                test_gen = test_generator("shots/testObject/testObject/test-"+str(i)+"-"+str(j)+"/", 1, img_size, name = name)
                results = model.predict_generator(test_gen,1,verbose=1)
                save_results("predicts/", results, name = name)
            except:
                continue
            