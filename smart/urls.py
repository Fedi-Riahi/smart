from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.views.generic.base import RedirectView

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', RedirectView.as_view(url='/login/', permanent=False), name='root'),  # Redirect root to login
    path('', include('cvApp.urls')),  # Include cvApp URLs
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
