from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def save(request):
    if request.method == 'POST':
        image = request.body
        print image
        with open('images/static/images/img/person.jpeg', 'w') as f:
            for chunk in image:
                f.write(chunk)
    return JsonResponse({'status':'success','message':'kthxbye. PS: We love you :)'})

def who(request):
    pass
