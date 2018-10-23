from django.urls import path, re_path
from ctc.api.views import UploadFile, GetResultByRequestIDAPIView, list_debug_directory
from rest_framework.schemas import get_schema_view

schema_view = get_schema_view(title='FlaxScanner API')

urlpatterns = [
    re_path(r'^v1/upload/$', UploadFile.as_view()),
    re_path(r'^v1/result/(?P<task_id>[0-9a-z-]+)/$', GetResultByRequestIDAPIView.as_view()),
    re_path(r'^v1/directory/(?P<task_id>[0-9a-z-]+)/', list_debug_directory)
]
