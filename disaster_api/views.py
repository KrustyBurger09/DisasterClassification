from django.shortcuts import render
import requests
import json
from django.views.decorators.csrf import csrf_exempt
from .text_classifier_api import text_classify
from .image_classifier_api import image_classify
from .maildemo import send_email_w_attachment
from .email_config import to, subject


from django.http import HttpResponse, HttpRequest


def index(request):
    return HttpResponse("Hello, world. You're at the index.")

@csrf_exempt
def classfiy_disaster(request):
    data = json.loads(request.body.decode('utf-8'))

    if data["text"]:
        print(data["text"])
        if (text_classify(data["text"])):
            body = "Disaster Tweet - " + str(data["text"])
            send_email_w_attachment(to, subject, body)
            return HttpResponse("This Post is related to disaster - " + str(data["text"]))
    elif data["image_path"]:
        print(data["image_path"])
        type_of_disaster = image_classify(data["image_path"])
        if (not type_of_disaster == "Non_Damage"):
            body = "Following Image is a " + str(type_of_disaster)
            send_email_w_attachment(to, subject, body, data["image_path"])
            return HttpResponse("This post is related to disaster - " + str(type_of_disaster))

    return HttpResponse("Not a Disaster Post")