from django.urls import path
from core import views

urlpatterns = [
    # Authentication & Shared
    path('', views.home_redirect, name='home'),
    path('login/', views.user_login, name='login'),
    path('logout/', views.user_logout, name='logout'),
    
    # User/Customer Portal
    path('profile/', views.user_profile, name='user_profile'),
    path('my-consumption/', views.user_consumption, name='user_consumption'),
    path('my-graphs/', views.user_graphs, name='user_graphs'),
    path('submit-feedback/', views.user_feedback, name='user_feedback'),
    
    # Admin Portal
    path('dashboard/', views.admin_dashboard, name='dashboard'),
    path('upload/', views.dataset_upload, name='upload_dataset'),
    path('training/', views.model_training, name='model_training'),
    path('predictions/', views.predictions_log, name='predictions'),
    path('fraud-alerts/', views.fraud_alerts, name='fraud_alerts'),
    path('graphs/', views.graph_analysis, name='graphs'),
    path('feedback/', views.admin_feedback_list, name='admin_feedback'),
    path('feedback/<int:feedback_id>/', views.admin_feedback_detail, name='admin_feedback_detail'),
    path('investigations/', views.investigations_list, name='investigations'),
    path('reports/', views.admin_reports, name='reports'),
    
    # Customer Details & Actions
    path('customers/', views.customer_list, name='customers'),
    path('customers/<str:customer_id>/', views.customer_detail, name='customer_detail'),
    path('customers/<str:customer_id>/block/', views.toggle_block_customer, name='toggle_block'),
]
