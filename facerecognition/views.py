import os
import time

from django.shortcuts import render
from rest_framework import generics, status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import FileUploadParser

from .models import employes
from .serializers import employeSerializer

from face_recognition_demo.face_recognition_demo import Visualizer, main

# Create your views here.

visualizer = Visualizer()
class recognition(APIView):
    parser_class = (FileUploadParser,)

    def post(self, request):
        client_data = employes(employe_photo = request.FILES['photo'])
        client_data.save()

        s = str(client_data.employe_photo)

        ori_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(ori_path, 'media', s)

        st = time.time()
        self.employe_name = str(main(visualizer, False, path, ""))
        print("Resutl time = " + str(time.time() - st))
        client_data.employe_name = self.employe_name

        print(self.employe_name.capitalize())
        client_data = employes.objects.get(employe_id=self.employe_name)
        print(client_data.employe_id)
        print(client_data.employe_name)
        print(client_data.employe_photo)

        clientSerializer = employeSerializer(data={'employe_name': client_data.employe_name,
                                                'employe_id': client_data.employe_id})
        os.remove(path)
        
        if self.employe_name == 'More than 1 faces found!':
            return Response({'error':self.employe_name}, status=status.HTTP_400_BAD_REQUEST)
        
        if self.employe_name == 'No face detected!':
            return Response({'error':self.employe_name}, status=status.HTTP_400_BAD_REQUEST)
        
        # if self.employe_name == 'Head is not adjusted correctly!':
        #     return Response({'error':self.employe_name}, status=status.HTTP_400_BAD_REQUEST)
        
        if self.employe_name == 'Unkown!':
            return Response({'error':self.employe_name}, status=status.HTTP_400_BAD_REQUEST)
        
        if clientSerializer.is_valid():
            return Response(clientSerializer.data, status=status.HTTP_201_CREATED)
        else:
            return Response(clientSerializer.errors, status=status.HTTP_400_BAD_REQUEST)

class add_employe(APIView):
    parser_class = (FileUploadParser,)

    def post(self, request):
        client_data = employes(employe_photo = request.FILES['photo'], employe_name = request.POST['name'], employe_id = request.POST['id'] )
        client_data.save()
        
        s = str(client_data.employe_photo)
        print("s:" + s)

        ori_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(ori_path, 'media', s)
        print(client_data.employe_name)
        print(client_data.employe_id)

        st = time.time()
        self.employe_name = str(main(visualizer, True, path, client_data.employe_id))
        print("Resutl time = " + str(time.time() - st))

        # client_data.employe_id = employes.objects.get(employe_name=self.employe_name)

        clientSerializer = employeSerializer(data={'employe_name': client_data.employe_name,
                                                'employe_id': client_data.employe_id})
        
        os.remove(path)
        
        if self.employe_name == 'More than 1 faces found!':
            return Response({'error':self.employe_name}, status=status.HTTP_400_BAD_REQUEST)
        
        if self.employe_name == 'No face detected!':
            return Response({'error':self.employe_name}, status=status.HTTP_400_BAD_REQUEST)
        
        if self.employe_name == 'Head is not adjusted correctly!':
            return Response({'error':self.employe_name}, status=status.HTTP_400_BAD_REQUEST)
        
        if self.employe_name == 'Unkown!':
            return Response({'error':self.employe_name}, status=status.HTTP_400_BAD_REQUEST)
        
        if clientSerializer.is_valid():
            # clientSerializer.save()
            return Response(clientSerializer.data, status=status.HTTP_201_CREATED)
        else:
            return Response(clientSerializer.errors, status=status.HTTP_400_BAD_REQUEST)

class employesListView(generics.ListAPIView):
    queryset = employes.objects.all()
    serializer_class = employeSerializer
