from resizeAllInDirectory import resizeToGivenSize
import os,sys

source="./Messi/"
dest="./Resized/"
dirs = os.listdir( source )

for file in dirs:
	print file
	resizeToGivenSize(250,250, source, file, dest)
