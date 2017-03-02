from sklearn.datasets import fetch_lfw_people
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import time 

#Original image size is 125 x 94, resizing to 0.4 gives 50 x 37
#Can also change the other parameter to have more classes but less accuracy
lfwPeople = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

nSamples, h, w = lfwPeople.images.shape

X = lfwPeople.data
nFeatures = X.shape[1]

Y = lfwPeople.target
targetNames = lfwPeople.target_names
nClasses = targetNames.shape[0]

print "Dataset Summary: " 
print "No. of samples - ",nSamples 
print "Height - ",h
print "Width - ",w
print "Features in Dataset - ",nFeatures
print "No. of class labels - ", nClasses


trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.25)

nComponents = 150

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


print "Predicting people's names on the test set"
startTime = time.time()
predictions = clf.predict(testXPCA)
print "Completed in ",(time.time() - startTime)

print (classification_report(testY, predictions, target_names=targetNames))
print (confusion_matrix(testY, predictions, labels=range(nClasses)))



