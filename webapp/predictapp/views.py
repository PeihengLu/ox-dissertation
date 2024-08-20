from django.shortcuts import render
from typing import Tuple

# Create your views here.
import json
import torch
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import pandas as pd
import numpy as np

import logging
log = logging.getLogger(__name__)

# Define your PyTorch model (replace with your actual model)
class DummyModel(torch.nn.Module):
    def forward(self, x):
        return torch.tensor([x.sum()])

model = DummyModel()

@csrf_exempt
def predict(request):
    log.info('Predict request received')
    if request.method == 'POST':
        log.info('POST request received')
        data = json.loads(request.body)
        sequence: str = data.get('dna_sequence', 0)
        pe_cell_line: str = data.get('pe_cell_line', 0)
        
        pe, cellline = pe_cell_line.split('-')
        
        pam_table = {
            'pe2': 'NGG',
        }
        
        
        trained_on_pridict_only = ['k562', 'adv']

        wt_sequence, mut_sequence, edit_position, mut_type, edit_length = prime_sequence_parsing(sequence)

        # return the pegRNA design in std format
        pegRNAs = propose_pegrna(sequence, mut_sequence, cellline in trained_on_pridict_only, edit_length=1, pam=pam_table[pe], pridict_only=cellline in trained_on_pridict_only)
        
        # load all models trained on the specified cell line and prime editors
        # then takes an average of the predictions

        return JsonResponse(pegRNAs)
    else:
        log.error('Invalid request method')
        return JsonResponse({'error': 'Invalid request method'}, status=400)
    
def prime_sequence_parsing(sequence: str) -> Tuple[str, str, int, int, int]:
    """
    Parse the sequence to extract the prime editing information

    Args:
        sequence (str): DNA sequence inputted by the user

    Returns:
        Tuple[str, str, int, int, int]: Tuple containing the wild type and mutated DNA sequence, edit position, mutation type, and edit length
    """
    pre_edit_sequence = sequence.split('(')[0]
    post_edit_sequence = sequence.split(')')[1]
    edit_position = len(pre_edit_sequence)
    wt_edit = sequence[edit_position+1:]
    wt_edit = wt_edit.split('/')[0]
    edit_length = len(wt_edit)
    mut_edit = sequence.split('/')[1][:edit_length]
    if '-' in mut_edit: # deletion
        mut_type = 2
    elif '-' in wt_edit: # insertion
        mut_type = 1
    else: # substitution
        mut_type = 0

    wt_sequence = pre_edit_sequence + wt_edit + post_edit_sequence
    mut_sequence = pre_edit_sequence + mut_edit + post_edit_sequence

    return wt_sequence, mut_sequence, edit_position, mut_type, edit_length



def propose_pegrna(wt_sequence: str, mut_sequence: str, edit_position: int, mut_type: int, edit_length: int, pam: str, pridict_only: bool) -> pd.DataFrame:
    pbs_len_range = np.range(8, 18) if not pridict_only else [13] 
    lha_len_range = range(0, 13)
    rha_len_range = range(7, 12)
    
    # in the range of lha length, scan for PAM sequences
    # edit must start before 3bp upstream of the PAM
    edit_to_pam_range = lha_len_range + 3
    
    protospacer_location_l = []
    protospacer_location_r = []
    pbs_location_l = []
    pbs_location_r = []
    lha_location_l = []
    lha_location_r = []
    rha_location_wt_l = []
    rha_location_wt_r = []
    rha_location_mut_l = []
    rha_location_mut_r = []
    rtt_location_wt_l = []
    rtt_location_wt_r = []
    rtt_location_mut_l = []
    rtt_location_mut_r = []
    sp_cas9_score = []
    mut_type = []
    # 99bp sequence starting from 10bp upstream of the protospacer
    wt_sequence = []
    mut_sequence = []
    
    
    for distance_to_pam in edit_to_pam_range:
        pass
    
    # return dummy data
    for i in range(10):
        protospacer_location_l.append(10)
        protospacer_location_r.append(20)
        pbs_location_l.append(5)
        pbs_location_r.append(8)
        lha_location_l.append(0)
        lha_location_r.append(3)
        rha_location_wt_l.append(20)
        rha_location_wt_r.append(25)
        rha_location_mut_l.append(20)
        rha_location_mut_r.append(25)
        rtt_location_wt_l.append(25)
        rtt_location_wt_r.append(30)
        rtt_location_mut_l.append(25)
        rtt_location_mut_r.append(30)
        sp_cas9_score.append(0.9)
        mut_type.append(0)
        wt_sequence.append('wt_sequence')
        mut_sequence.append('mut_sequence')
    
    
    df = pd.DataFrame({
        'protospacer_location_l': protospacer_location_l,
        'protospacer_location_r': protospacer_location_r,
        'pbs_location_l': pbs_location_l,
        'pbs_location_r': pbs_location_r,
        'lha_location_l': lha_location_l,
        'lha_location_r': lha_location_r,
        'rha_location_wt_l': rha_location_wt_l,
        'rha_location_wt_r': rha_location_wt_r,
        'rha_location_mut_l': rha_location_mut_l,
        'rha_location_mut_r': rha_location_mut_r,
        'rtt_location_wt_l': rtt_location_wt_l,
        'rtt_location_wt_r': rtt_location_wt_r,
        'rtt_location_mut_l': rtt_location_mut_l,
        'rtt_location_mut_r': rtt_location_mut_r,
        'sp_cas9_score': sp_cas9_score,
        'mut_type': mut_type,
        'wt_sequence': wt_sequence,
        'mut_sequence': mut_sequence,
    })
    json_data = df.to_json(orient='records')
    return json_data
    
    

def index(request):
    log.info('Index request received')
    return render(request, 'predictapp/index.html')