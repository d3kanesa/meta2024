from django.urls import path
from pointGen import views

urlpatterns = [
    path('', views.home_view, name='home'),
    path('video_feed/', views.video_feed, name='video_feed'),
    path('hand-keypoints/', views.hand_keypoints_view, name='hand_keypoints'),
    # path('pose-detection/', views.pose_detection_view, name='pose_detection'),
]