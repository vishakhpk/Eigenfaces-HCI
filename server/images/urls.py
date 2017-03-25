from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'save/',views.save),
    url(r'who/', views.who)
]
