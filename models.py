
from django.db import models

class Video(models.Model):
    # title = models.CharField(max_length=255)
    # upload_date = models.DateTimeField(auto_now_add=True)
    video_file = models.FileField(upload_to='videos/')


class Video1(models.Model):
    video_file = models.FileField(upload_to='videos1/')



class Video2(models.Model):
    video_file = models.FileField(upload_to='videos2/')

