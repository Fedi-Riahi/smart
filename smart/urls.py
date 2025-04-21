from django.contrib import admin
from django.urls import include, path
from django.views.generic.base import RedirectView
from django.conf import settings
from django.conf.urls.static import static


# smart/urls.py
urlpatterns = [
    path('admin/', admin.site.urls),
    path('accounts/', include('django.contrib.auth.urls')),
    path('', include('cvApp.urls')),
    path('', RedirectView.as_view(url='/upload/', permanent=False)),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
