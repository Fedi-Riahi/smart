# smart/cvApp/serializers.py
from rest_framework import serializers
from .models import CV

class CVSerializer(serializers.ModelSerializer):
    class Meta:
        model = CV
        fields = ['user', 'file', 'extracted_text', 'extracted_data', 'modified_file', 'uploaded_at']  # Adjust fields as needed
