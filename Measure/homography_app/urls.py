from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload_image, name='upload_image'),  # Home page for image upload
    path('measure-distance/', views.measure_distance_view, name='measure_distance'),  # Measure distance page
    path('mark_distance/', views.mark_distance, name='mark_distance'),
    path('mark_distance_image/', views.mark_distance_image, name='mark_distance_image'),
    path('upload_video/', views.upload_video, name='upload_video'),

]

