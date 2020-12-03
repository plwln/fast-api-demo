
import os
from PIL import Image

def resize(type, path_to_save):
    for pathName in os.listdir('withObjects/'):
        for file in os.listdir('withObjects/%s/'%pathName):
            if type in file:
                image = Image.open('withObjects/%s/%s'%(pathName,file))
                image = image.resize((256,256))
                image.save(path_to_save+'/%s'%file.replace('jpg','png'))

resize('mask','masksNew')
resize('color','colorFiles')