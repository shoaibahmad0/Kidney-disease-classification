from django import forms


class ImageUploadForm(forms.Form):
    image = forms.ImageField(label='Upload Kidney Image')

    def clean_image(self):
        image = self.cleaned_data['image']
        if image.size > 50 * 1024 * 1024:
            raise forms.ValidationError('Images must be 50 MB or smaller.')
        return image


from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User

class CustomUserCreationForm(UserCreationForm):
    email = forms.EmailField(required=True)

    class Meta:
        model = User
        fields = ("username", "email", "password1", "password2")
