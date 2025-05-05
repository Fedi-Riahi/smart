from django.urls import path
from . import views
from django.contrib.auth.views import LoginView, LogoutView

urlpatterns = [
    path('upload/', views.upload_cv, name='upload_cv'),
    path('results/<int:cv_id>/', views.cv_results, name='cv_results'),
    path('modify/<int:cv_id>/', views.modify_cv, name='modify_cv'),
     path('modify-section/<int:cv_id>/', views.modify_cv_section, name='modify_cv_section'),
    path('download/<int:cv_id>/', views.download_modified_cv, name='download_modified_cv'),
    path('login/', LoginView.as_view(template_name='login.html'), name='login'),
    path('logout/', LogoutView.as_view(next_page='login'), name='logout'),
]
