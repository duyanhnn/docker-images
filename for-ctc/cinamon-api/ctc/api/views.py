import os
import uuid

from django.conf import settings
from celery.result import AsyncResult
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.generics import CreateAPIView
from rest_framework.parsers import MultiPartParser, FormParser
from ctc.utils.file import read_pdf_image
# from django.core.files.storage import FileSystemStorage
from ctc.serializer.file_upload import FileUploadSerializer, CSVUploadSerializer
from ctc.utils.response import to_json, get_absolute_url
from ctc.model.task import Task
from django.shortcuts import render
from ctc.tasks.celery import app, process_pdf


class UploadFile(CreateAPIView):
    serializer_class = FileUploadSerializer
    parser_classes = (MultiPartParser, FormParser, )

    def post(self, request):
        uploaded_file = request.data['image']
        request_id = request.data['request_id']
        import re
        if not re.match(r'^[A-Za-z0-9-]+$', request_id):
            return to_json(data="Request ID can only contain letters, numbers and hyphens",
                          status_code=status.HTTP_400_BAD_REQUEST, status=0)
        print("FLAX ------ Received request ID: ", request_id)
        # bad request
        if not request_id:
            return to_json(data="Please specify request id for checking result later.",
                           status_code=status.HTTP_400_BAD_REQUEST, status=0)
        if not uploaded_file:
            return to_json(data="Please choose file to upload", status_code=status.HTTP_400_BAD_REQUEST, status=0)
        file_ext = os.path.splitext(uploaded_file.name)[-1].lower()
        if not file_ext:
            return to_json(data="Uploaded file doesn't have extension.", status_code=status.HTTP_400_BAD_REQUEST, status=0)
        if file_ext not in ['.pdf']:
            return to_json(data="Only allow .pdf files", status_code=status.HTTP_400_BAD_REQUEST, status=0)

        task_id = str(request_id).strip()
        process_id = str(uuid.uuid4())
        file_name = uploaded_file.name
        new_file_name = '{}{}'.format(task_id, file_ext)
        pdf_file_path = os.path.join(settings.UPLOAD_FILE_PATH, new_file_name)
        # from django.core.files.storage import FileSystemStorage
        # fs = FileSystemStorage()
        # pdf_file_path = fs.save(new_file_name, uploaded_file)
        with open(pdf_file_path, 'wb') as f:
            for chunk in uploaded_file.chunks():
                f.write(chunk)
        print("FLAX ------ Saved file: ", pdf_file_path)
        img_from_pdf_file = os.path.join(settings.UPLOAD_FILE_PATH, '{}.png'.format(task_id))
        read_pdf_image(pdf_file_path, img_from_pdf_file)
        # img_from_pdf_file = os.path.join(os.path.dirname(pdf_file_path), '{}.png'.format(task_id))
        new_file_name = os.path.basename(img_from_pdf_file)
        task = Task(request_id=task_id, id=process_id, image=new_file_name, file_name=file_name)
        print("FLAX ------ Created task: ", task_id)
        try:
            task.save()
            process_pdf.apply_async((task_id,), task_id=task_id)
            debug = {
                "result_url": '{}/api/v1/result/{}/'.format(get_absolute_url(request), task_id)
            }
        except Exception as e:
            print("FLAX ------: ", e)
            return to_json(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, request_id=task_id, process_id=process_id,
                           file_name=file_name, status=0, data=str(e))
        return to_json(status_code=status.HTTP_200_OK, request_id=task_id, process_id=process_id, file_name=file_name, status=0, debug=debug)


class GetResultByRequestIDAPIView(APIView):
    serializer_class = CSVUploadSerializer
    parser_classes = (MultiPartParser, FormParser,)

    def get(self, request, task_id):
        request_id = task_id
        try:
            task = Task.objects.get(pk=request_id)
        except ValueError as e:
            return to_json(data="Invalid request id!", status_code=status.HTTP_404_NOT_FOUND, status=0)
        image_file = settings.UPLOAD_FILE_PATH + str(task.image)

        if task.data:
            error = task.data.get('error')
            if error:
                return to_json(error=error)
            else:
                import ast
                data_dict = ast.literal_eval(str(task.data))
                return to_json(request_id=request_id, process_id=task.id, file_name=task.file_name, data=data_dict["data"])
        else:
            # result = AsyncResult(id=request_id, app=app)
            state = 'PENDING'
            if state == 'PENDING':
                state = 'RUNNING'
            return to_json(status_code=status.HTTP_200_OK, status=state, file_name=task.file_name, process_id=task.id, request_id=request_id)


def list_debug_directory(request, task_id):
    debug_directory = os.path.join(settings.DEBUG_FILE_PATH, task_id)
    file_list = sorted(os.listdir(debug_directory))
    return render(request, 'list_file.html', {'files': file_list, 'task_id': task_id})
