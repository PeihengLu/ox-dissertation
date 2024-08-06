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
        sequence: str = data.get('dna_sequence', 0)
        pe_cell_line: str = data.get('pe_cell_line', 0)
        
        pe, cellline = pe_cell_line.split('-')
        
        pam_table = {
            'pe2': 'NGG',
        }
        
        trained_on_pridict_only = ['k562', 'adv']
        
        sequences, pegRNAs = propose_pegrna(sequence, pam_table[pe], cellline in trained_on_pridict_only)
        
        # load all models trained on the specified cell line and prime editors
        # then takes an average of the predictions

        # parse user's input
        return JsonResponse({'prediction': prediction})
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=400)

def propose_pegrna(dna_sequence: str, pam: str, pridicrt_only: bool) -> str:
    pbs_len_range = range(8, 18) if not pridicrt_only else [13] 
    lha_len_range = range(0, 13)
    rha_len_range = range(7, 12)
    

def index(request):
    return render(request, 'predictapp/index.html')