from rest_framework import serializers
from ctc.model.task import Task


class FileUploadSerializer(serializers.HyperlinkedModelSerializer):
    
    class Meta:
        model = Task
        fields = ("image", "request_id", "created_at")


class CSVUploadSerializer(serializers.Serializer):
    csv = serializers.FileField()