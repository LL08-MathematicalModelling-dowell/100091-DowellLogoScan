from django.urls import path
from .views import logo_upload_view, logo_upload_view2

urlpatterns = [
    path('upload-logo-segmentation', logo_upload_view, name='upload-logo'),
    path('upload-logo-nosegmentation', logo_upload_view2, name='upload-logo2'),
    
]
