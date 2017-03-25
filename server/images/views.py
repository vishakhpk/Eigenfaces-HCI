import io
import json
import Image
import base64

import numpy as np
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

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

        with open('images/static/images/img/person.jpeg', 'w') as f:
            a_cropped.save(f)
    return JsonResponse({'status':'success','message':'kthxbye. PS: We love you :)'})

@csrf_exempt
def who(request):
    pass
