from django.db import models
from django_mysql.models import JSONField


class Task(models.Model):
    id = models.UUIDField()
    request_id = models.TextField(primary_key=True)
    file_name = models.TextField()
    image = models.FileField()
    data = JSONField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'task'
