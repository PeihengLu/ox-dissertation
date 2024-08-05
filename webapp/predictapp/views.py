from django.shortcuts import render

# Create your views here.
import json
import torch
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

# Define your PyTorch model (replace with your actual model)
class DummyModel(torch.nn.Module):
    def forward(self, x):
        return torch.tensor([x.sum()])

model = DummyModel()

@csrf_exempt
def predict(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        user_input: str = data.get('input', 0)

        # parse user's input

        input_tensor = torch.tensor([user_input], dtype=torch.float32)
        prediction = model(input_tensor).item()
        return JsonResponse({'prediction': prediction})
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=400)


def index(request):
    return render(request, 'predictapp/index.html')