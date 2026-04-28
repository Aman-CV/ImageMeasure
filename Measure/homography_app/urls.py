from django.urls import path, re_path
from . import views

urlpatterns = [
    path('', views.landing_page, name='landing_page'),
    path('upload_video/', views.upload_video, name='upload_video'), #upload video from app
    path('list_videos/', views.list_videos), # get all videos
    path('calibrate/',views.upload_calibration_video), #calibrate video and generate homo matrix
    #path('calibration_info/', views.get_homograph), #get calib info
    path('get_processed_video/', views.get_video_detail), #get processed video
    path('get_processed_video_link/', views.get_video_link), #get processed video

    re_path(r'^stream_video/(?P<video_id>\d+)/?$', views.video_stream, name='video_stream'),
            #path('process_image/', views.process_image, name='process_image'),
    path('list_videos_by_assessment_and_test/', views.list_videos_by_assessment_and_test,
         name='list_videos_by_assessment_and_test'),
]

