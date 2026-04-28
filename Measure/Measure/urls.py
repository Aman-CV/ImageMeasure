from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    # original
    path('admin/', admin.site.urls),
    path('', include('homography_app.urls')),
    path('imagemeasure/', include('homography_app.urls')),
]