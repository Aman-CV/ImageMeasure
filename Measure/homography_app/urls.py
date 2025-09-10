from django.urls import path
from . import views

urlpatterns = [
    path('upload_video/', views.upload_video, name='upload_video'), #upload video from app
    path('list_videos/', views.list_videos), # get all videos
    path('calibrate/',views.upload_calibration_video), #calibrate video and generate homo matrix
    path('calibration_info/', views.get_homograph), #get calib info
    #path('get_processed_video/', blah blah), #get processed video
]

