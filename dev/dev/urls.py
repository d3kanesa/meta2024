from django.urls import path
from pointGen import views
from django.views.generic import TemplateView

urlpatterns = [
    path('', views.home_view, name='home'),
    path('video_feed/', views.video_feed, name='video_feed'),
    path('hand-keypoints/', views.hand_keypoints_view, name='hand_keypoints'),
    path('api/hand-keypoints/', views.handle_recording, name='handle_recording'),  # New API endpoint
    path('saarujan/', TemplateView.as_view(template_name='saarujan.html'), name='saarujan'),  # New TemplateView

]