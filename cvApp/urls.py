from django.urls import path
from . import views

urlpatterns = [
    path('upload/', views.upload_cv, name="upload_cv"),
    path('results/<int:cv_id>/', views.cv_results, name='cv_results'),
]
