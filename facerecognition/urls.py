from django.urls import path
from . import views

app_name = 'facerecognition'

urlpatterns = [
    path('check_employes/', views.recognition.as_view(), name="recognition"),
    path('add_employes/', views.add_employe.as_view(), name="add_employe"),
    path('get_employes/', views.employesListView.as_view(), name="get_all"),
]