from django.db import models

# Create your models here.

class Logo(models.Model):
  image = models.ImageField(blank=False, null=False, upload_to="uploaded_logos")
  timestamp = models.DateTimeField(auto_now_add=True)

  def __str__(self) -> str:
    return self.image.name