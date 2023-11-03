
from rest_framework import serializers
from .models import Video,Video1,Video2

class VideoSerializer(serializers.ModelSerializer):
    class Meta:
        model = Video
        fields = '__all__'

class VideoSerializer1(serializers.ModelSerializer):
    class Meta:
        model = Video1
        fields = '__all__'

class VideoSerializer2(serializers.ModelSerializer):
    class Meta:
        model = Video2
        fields = '__all__'

