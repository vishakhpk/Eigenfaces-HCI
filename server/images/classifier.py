from sklearn.datasets import fetch_lfw_people
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from PIL import Image
import numpy as np
import pickle
import time
import os

def train(directory):
	lfwPeople = fetch_lfw_people(min_faces_per_person=50, resize=0.4)
	nSamples, h, w = lfwPeople.images.shape
	X = lfwPeople.data
	Y = lfwPeople.target
	targetNames = lfwPeople.target_names

	image_names = os.listdir('images/static/images/'+directory+'/')
	images = []
	for image in image_names:
	    images.append(Image.open('images/static/images/'+directory+'/'+image))
	    images_array = []
	for image in images:
	    image_array = np.array(image.convert('L').resize((37, 50)))
	    print image_array.shape
	    images_array.append(image_array.ravel())
	images_array = np.asarray(images_array)
	new_labels = [targetNames.shape[0] for image in images]
	print new_labels

	npX = np.vstack((X,images_array))
	npY = np.concatenate((Y,new_labels))
	
	nFeatures = npX.shape[1]
	targetNames = np.append(targetNames,directory)
	nClasses = targetNames.shape[0]


	trainX, testX, trainY, testY = train_test_split(npX, npY, test_size=0.25)

	nComponents = 500

	print "Extracting the top ",nComponents," components from ",trainX.shape[0]," faces"
	startTime = time.time()
	pca = PCA(n_components=nComponents, whiten=True).fit(trainX)
	print "Completed in ", (time.time() - startTime)

	eigenfaces = pca.components_.reshape((nComponents, h, w))

	print("Projecting the input data on the eigenfaces orthonormal basis")
	startTime = time.time()
	trainXPCA = pca.transform(trainX)
	testXPCA = pca.transform(testX)
	print "Completed in ",(time.time() - startTime)


	print("Fitting the SVM to the training set")
	startTime = time.time()
	param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
	              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
	clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
	clf = clf.fit(trainXPCA, trainY)

	print "Completed in ",(time.time() - startTime)
	print "Best estimator found by grid search:"
	print clf.best_estimator_
	with open('classifier.pkl','w') as f:
		pickle.dump({'clf':clf,'pca':pca},f)
	print "Predicting people's names on the test set"
	startTime = time.time()
	predictions = clf.predict(testXPCA)
	print "Completed in ",(time.time() - startTime)

	print (classification_report(testY, predictions, target_names=targetNames))
	print (confusion_matrix(testY, predictions, labels=range(nClasses)))

if __name__ == '__main__':
	train('person01')