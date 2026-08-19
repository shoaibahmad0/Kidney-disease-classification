from io import BytesIO
from types import SimpleNamespace

from django.contrib.auth.models import User
from django.core.exceptions import ValidationError
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase
from django.urls import reverse
from PIL import Image

from .forms import ImageUploadForm
from .models import Prediction


class ImageUploadFormTests(TestCase):
    def test_rejects_files_larger_than_50_mb(self):
        form = ImageUploadForm()
        form.cleaned_data = {
            'image': SimpleNamespace(size=50 * 1024 * 1024 + 1),
        }

        with self.assertRaisesMessage(ValidationError, 'Images must be 50 MB or smaller.'):
            form.clean_image()


class PredictionAccessTests(TestCase):
	def setUp(self):
		self.owner = User.objects.create_user(username='owner', password='pass1234')
		self.other_user = User.objects.create_user(username='other', password='pass1234')
		self.prediction = Prediction.objects.create(
			user=self.owner,
			patient_name='Test Patient',
			image=SimpleUploadedFile('scan.jpg', self._image_bytes(), content_type='image/jpeg'),
			image_name='scan.jpg',
			prediction='Normal',
			confidence=99.0,
			all_confidences=[99.0, 1.0],
			time_taken=0.1,
		)

	@staticmethod
	def _image_bytes():
		image = Image.new('RGB', (2, 2), color='white')
		output = BytesIO()
		image.save(output, format='JPEG')
		return output.getvalue()

	def test_delete_requires_prediction_owner(self):
		self.client.force_login(self.other_user)

		response = self.client.post(reverse('delete_prediction', args=[self.prediction.id]))

		self.assertEqual(response.status_code, 404)
		self.assertTrue(Prediction.objects.filter(pk=self.prediction.pk).exists())

	def test_delete_allows_prediction_owner(self):
		self.client.force_login(self.owner)

		response = self.client.post(reverse('delete_prediction', args=[self.prediction.id]))

		self.assertRedirects(response, reverse('history'))
		self.assertFalse(Prediction.objects.filter(pk=self.prediction.pk).exists())

# Create your tests here.
