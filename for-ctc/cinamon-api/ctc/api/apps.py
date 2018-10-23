from django.apps import AppConfig
from django.conf import settings


class ApiConfig(AppConfig):
    name = 'api'

    def ready(self):
        # disable loading models in uwsgi/API app
        settings.LOAD_MODEL = False
