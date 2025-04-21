from django.db import models
from django.contrib.auth.models import User

class CV(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    file = models.FileField(upload_to='cvs/')
    extracted_data = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"CV {self.id} by {self.user.username}"
