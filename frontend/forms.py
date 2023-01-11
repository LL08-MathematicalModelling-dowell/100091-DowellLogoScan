from django import forms
from backend.models import Logo

class LogoUploadForm(forms.ModelForm):
    class Meta:
        model = Logo
        fields = ['image',]
