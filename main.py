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
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

def unzip(name, url):
    filehandle, _ = urllib.urlretrieve(url)
    zip_file_object = zipfile.ZipFile(filehandle, 'r')
    try:
        os.mkdir("shots/"+name)
    except:
        shutil.rmtree(name, ignore_errors=True)
    zip_file_object.extractall('shots/'+name+"/")
    for n in os.listdir('shots/'+name+"/"):
        image = Image.open('shots/'+name+"/"+n)
        image.mode = 'I'
        try:
            image.point(lambda i: i*(1./256)).convert('L').save('shots/'+name+"/"+n.replace(".tif","")+'.jpeg', "JPEG")
        except:
            continue


class Item(BaseModel):
    urls: dict

class imageRequest(BaseModel):
    name: str
    band: str
    type: str

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
        unzip(name, item.urls[name])


@app.post("/api/get_image")
async def image_endpoint(images: imageRequest):
    # Returns a cv2 image array from the document vector
    # img = Image.open('./'+images.name+"/image.B2.jpeg")
    path = [n for n in os.listdir('shots/'+images.name+"/") if images.band in n and images.type in n]
    return FileResponse('shots/'+images.name+"/"+path[0])
