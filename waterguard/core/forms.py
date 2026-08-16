from django import forms
from core.models import DatasetUpload, Feedback, Investigation, UserProfile

class DatasetUploadForm(forms.ModelForm):
    class Meta:
        model = DatasetUpload
        fields = ['file']
        widgets = {
            'file': forms.ClearableFileInput(attrs={'class': 'form-control', 'accept': '.csv,.xlsx,.xls'}),
        }

class FeedbackForm(forms.ModelForm):
    class Meta:
        model = Feedback
        fields = ['feedback_type', 'message']
        widgets = {
            'feedback_type': forms.Select(attrs={'class': 'form-select'}),
            'message': forms.Textarea(attrs={'class': 'form-control', 'rows': 4, 'placeholder': 'Describe your meter or consumption issue here...'}),
        }

class FeedbackResponseForm(forms.ModelForm):
    class Meta:
        model = Feedback
        fields = ['response', 'status']
        widgets = {
            'response': forms.Textarea(attrs={'class': 'form-control', 'rows': 3, 'placeholder': 'Type response to customer...'}),
            'status': forms.Select(attrs={'class': 'form-select'}),
        }

class InvestigationForm(forms.ModelForm):
    class Meta:
        model = Investigation
        fields = ['status', 'remarks', 'confirmed_fraud']
        widgets = {
            'status': forms.Select(attrs={'class': 'form-select'}),
            'remarks': forms.Textarea(attrs={'class': 'form-control', 'rows': 4, 'placeholder': 'Enter inspection details and actions taken...'}),
            'confirmed_fraud': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        }

class UserProfileForm(forms.ModelForm):
    class Meta:
        model = UserProfile
        fields = ['phone', 'address']
        widgets = {
            'phone': forms.TextInput(attrs={'class': 'form-control'}),
            'address': forms.Textarea(attrs={'class': 'form-control', 'rows': 2}),
        }
