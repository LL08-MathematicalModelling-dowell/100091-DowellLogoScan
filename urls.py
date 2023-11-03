"""mysite URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from .views import *

urlpatterns = [
    path('', index),
    path('admin/', admin.site.urls),
    path('login/', AdminAuth.as_view(), name='admin_auth'),
    path('upload/', VideoUploadView.as_view(), name='video-upload'),
    path('uploadfast/', VideoUploadViewFastUpload.as_view(), name='video-uploadfast'),
    path('VideoUploadView_progress/',VideoUploadView_progress.as_view(),name='video-upload'),
    path('VideoUploadView1/',VideoUploadView1.as_view(),name='video-upload1'),
    path('VideoUploadViewFrames/',VideoUploadViewFrames.as_view(),name='VideoUploadViewFrames'),
    path('dropdown-menu-data/', DropDownMenuData.as_view(),name='dropdown-menu-data'),
    path('image/<str:id>', ImageAPIView.as_view(), name='image_api'),
    path('reviews/', ReviewsView.as_view(), name='reviews'),
    path('user-review/', UserReview.as_view(), name='user-review'),
    path('ImageComparision/', image_comparision, name='image_upload'),
    path('logo-upload-image/', LogoImageUploadView.as_view(), name='logo_upload'),
]
