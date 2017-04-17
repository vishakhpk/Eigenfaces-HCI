import io
import os
import sys
import json
import base64
import pickle
import classifier

import numpy as np
from models import Person
from PIL import Image
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt


clf = None
pca = None
with open('classifier.pkl') as f:
    classifier_components = pickle.load(f)
    clf = classifier_components['clf']
    pca = classifier_components['pca']

@csrf_exempt
def save(request):
    if request.method == 'POST':
        image = request.FILES['image']
        a_image = Image.open(image)
        x = int(7.2*float(request.POST['x']))
        y = int(7.2*float(request.POST['y']))
        w = int(7.3*float(request.POST['width']))
        h = int(7.3*float(request.POST['height']))
        ey = int(float(request.POST['eulerZ']))
        print x,y,w,h,ey
        a_cropped = a_image.crop( (x, y, x+w, y+h) )
        image_array = np.array(a_cropped.convert('L').resize((37, 50)))
        # predictions = clf.predict(pca.transform([image_array.ravel(), image_array.ravel()]))
        image_number = str(len(os.listdir('images/static/images/person01/'))).zfill(2)
        with open('images/static/images/person01/'+image_number+'.jpeg', 'w') as f:
            a_cropped.save(f)
    return JsonResponse({'status':'success','message':'kthxbye. PS: We love you :)','user':'123', 'userName':'Dont matter'})

@csrf_exempt
def who(request):
    clf = None
    pca = None
    with open('classifier.pkl') as f:
        classifier_components = pickle.load(f)
        clf = classifier_components['clf']
        pca = classifier_components['pca']
    if request.method == 'POST':
        image = request.FILES['image']
        a_image = Image.open(image)
        x = int(7.2*float(request.POST['x']))
        y = int(7.2*float(request.POST['y']))
        w = int(7.3*float(request.POST['width']))
        h = int(7.3*float(request.POST['height']))
        ey = int(float(request.POST['eulerZ']))
        print x,y,w,h,ey
        a_cropped = a_image.crop( (x, y, x+w, y+h) )
        image_array = np.array(a_cropped.convert('L').resize((37, 50)))
        predictions = clf.predict(pca.transform([image_array.ravel(), image_array.ravel()]))
        print predictions
    response = {'stuff':{'status':'success','message':'kthxbye. PS: We love you :)','user':str(predictions[0]), 'userName':Person.objects.get(person_id=predictions[0]).person_name, 'details':Person.objects.get(person_id=predictions[0]).person_details}}
    print response
    return JsonResponse(response)

@csrf_exempt
def train(request):
    classifier.train(request.POST['user'])
    return JsonResponse({'status':'success','message':'kthxbye. PS: We love you :)'})
