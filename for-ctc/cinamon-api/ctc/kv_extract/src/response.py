from collections import OrderedDict

from rest_framework import status
from rest_framework.response import Response

def to_json(data="", error="", version="1.0", debug="", status=status.HTTP_200_OK):
    message = OrderedDict([("data", ""), ("version", ""), ("error", "")])
    message['data'] = data
    message['error'] = error
    message['version'] = version
    if debug:
        message['debug'] = debug
    return Response(message, status=status)

def get_absolute_url(request):
    return request.build_absolute_uri('/').strip("/")