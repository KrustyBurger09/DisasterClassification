from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("classfiy_disaster/", views.classfiy_disaster, name="classfiy_disaster")
]