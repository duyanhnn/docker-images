from collections import OrderedDict
from rest_framework import status
from rest_framework.response import Response


def to_json(data="", process_id="", request_id="", file_name="", status=1, error="", version="1.0", debug="", status_code=status.HTTP_200_OK):
    message = OrderedDict([("status_code", ""), ("process_id", ""), ("status", "")])
    message['status_code'] = status_code
    message['request_id'] = request_id
    message['process_id'] = process_id
    message['file_name'] = file_name
    message['status'] = status
    if data:
        message['data'] = data
    if debug:
        message['debug'] = debug
    return Response(message, status=status_code)


def get_absolute_url(request):
    return request.build_absolute_uri('/').strip("/")