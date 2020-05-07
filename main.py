from fastapi import FastAPI
from pydantic import BaseModel
import urllib.request as urllib
import zipfile
import os
from PIL import Image
import io
from starlette.responses import StreamingResponse
from fastapi.responses import FileResponse
import shutil

def unzip(name, url):
    filehandle, _ = urllib.urlretrieve(url)
    zip_file_object = zipfile.ZipFile(filehandle, 'r')
    try:
        os.mkdir(name)
    except:
        shutil.rmtree(name, ignore_errors=True)
    zip_file_object.extractall('./'+name+"/")
    for n in os.listdir('./'+name+"/"):
        image = Image.open('./'+name+"/"+n)
        image.mode = 'I'
        try:
            image.point(lambda i: i*(1./256)).convert('L').save('./'+name+"/"+n.replace(".tif","")+'.jpeg', "JPEG")
        except:
            continue


class Item(BaseModel):
    urls: dict

class imageRequest(BaseModel):
    name: str
    band: str

app = FastAPI()

@app.post('/unzip')
def unzip_page(item: Item):
    for name in item.urls.keys():
        unzip(name, item.urls[name])


@app.post("/get_image")
async def image_endpoint(images: imageRequest):
    # Returns a cv2 image array from the document vector
    # img = Image.open('./'+images.name+"/image.B2.jpeg")
    path = [n for n in os.listdir('./'+images.name+"/") if images.band in n ]
    return FileResponse('./'+images.name+"/"+path[0])