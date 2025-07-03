from django.urls import path
from classifier import views
from django.contrib.auth import views as auth_views
from classifier.views import activate_account
from django.contrib.auth.views import LogoutView

urlpatterns = [
    path('', views.landing_redirect, name='landing'),
    path('signup/', views.signup_view, name='signup'),
    path('login/', auth_views.LoginView.as_view(template_name='login.html'), name='login'),
    path('logout/', LogoutView.as_view(next_page='login'), name='logout'),
    path('history/', views.history_view, name='history'),
    path('upload/', views.upload_image, name='upload'), 
    path('activate/<uidb64>/<token>/', activate_account, name='activate'),
]
