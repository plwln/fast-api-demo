from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
import os
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
    uvicorn.run("server:app", host="0.0.0.0", port=8082, log_level="trace")

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/api/get_image")
async def image_endpoint(images: imageRequest):
    print("Start")
    # Returns a cv2 image array from the document vector
    # img = Image.open('./'+images.name+"/image.B2.jpeg")
    folder = 'shots/'
    path = [n for n in os.listdir(folder+images.name+"/") if images.band in n and images.type in n]
    print(path[0])
    return FileResponse(folder+images.name+"/"+path[0])