
import os, sys
import PIL
from PIL import Image

def resizeAllJpegs(sourceDirectory,writeDirectory):
    dirs = os.listdir( sourceDirectory )
    #set to extremely large values. May be more efficient than finding size of first file in directory of type jpeg?
    smallestWidth = 20000000
    smallestHeight=20000000
    #find the smallest heights and widths
    for file in dirs:
        if file.endswith(".jpg") or file.endswith(".jpeg"):
            dims = getImageSize(sourceDirectory,file)
            if dims['width']<smallestWidth:
                smallestWidth = dims['width']
            if dims['height']<smallestHeight:
                smallestHeight = dims['height']

    for file in dirs:
       if file.endswith(".jpg") or file.endswith(".jpeg"):
           resizeToGivenSize(smallestWidth,smallestHeight,sourceDirectory,file,writeDirectory)


def resizeToGivenSize(width,height,fromPath,file,savePath):
    img = Image.open(''+fromPath+file)
    img = img.resize((width, height), PIL.Image.ANTIALIAS)
    img.save(savePath+'resized-'+file)

def getImageSize(path,file):
    img = Image.open(''+path+file)
    width, height = img.size
    print width , height
    return {'width':width,'height':height}
