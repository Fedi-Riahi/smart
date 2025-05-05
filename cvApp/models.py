from django.db import models
from django.contrib.auth.models import User

class CV(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    file = models.FileField(upload_to='cvs/')
    extracted_text = models.TextField(blank=True)  # Raw text from PDF
    extracted_data = models.JSONField(default=dict)  # Skills and suggestions
    modified_file = models.FileField(upload_to='modified_cvs/', null=True, blank=True)  # Modified PDF
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"CV {self.id} for {self.user.username}"
