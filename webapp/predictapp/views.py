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
        pe = pe.lower()
        cellline = cellline.lower()
        
        pam_table = {
        'pe2max_epegrna': 'NGG',
        'pe2max': 'NGG',
        'pe4max': 'NGG',
        'pe4max_epegrna': 'NGG',
        'nrch_pe4max': 'NGG',
        'pe2': 'NGG',
        'nrch_pe2': 'NGG',
        'nrch_pe2max': 'NGG',
    }
        
        
        trained_on_pridict_only = ['k562', 'adv']

        wt_sequence, mut_sequence, edit_position, mut_type, edit_length = prime_sequence_parsing(sequence)

        # return the pegRNA design in std format
        pegRNAs = propose_pegrna(wt_sequence=wt_sequence, mut_sequence=mut_sequence, edit_position=edit_position, mut_type=mut_type, edit_length=edit_length, pam=pam_table[pe], pridict_only=cellline in trained_on_pridict_only)
        
        # load all models trained on the specified cell line and prime editors
        # then takes an average of the predictions
        # TODO implement the model loading and prediction
        # TODO realign the locations to starting from 10 bp upstream of the protospacer
        pegRNAs['editing_efficiency'] = [0.1 for _ in range(len(pegRNAs))]

        # return the pegRNAs as well as the original sequence
        response = {
            'pegRNAs': pegRNAs.to_dict(orient='records'),
            'full_sequence': wt_sequence,
        }

        return JsonResponse(response, safe=False)
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
    pbs_len_range = np.arange(8, 18) if not pridict_only else [13] 
    lha_len_range = np.arange(0, 13)
    rha_len_range = np.arange(7, 20)
    
    # in the range of lha length, scan for PAM sequences
    # edit must start before 3bp upstream of the PAM
    edit_to_pam_range = lha_len_range + len(pam)
    
    protospacer_location_l = []
    protospacer_location_r = []
    pbs_location_l = []
    pbs_location_r = []
    lha_location_l = []
    lha_location_r = []
    rha_location_l = []
    rha_location_r = []
    rtt_location_l = []
    rtt_location_r = []
    sp_cas9_score = []
    mut_types = []
    # 99bp sequence starting from 10bp upstream of the protospacer
    wt_sequences = []
    mut_sequences = []
    
    for pam_distance_to_edit in edit_to_pam_range:
        # no valid PAM sequence
        # PAM is 3bp downstream of nicking site
        # nicking site is the end of PBS and start of LHA
        pam_position = edit_position - pam_distance_to_edit
        if not match_pam(wt_sequence[edit_position - pam_position: edit_position - pam_position + len(pam)] , pam):
            continue
        nicking_site = pam_position - 3
        for pbs_len in pbs_len_range:
            for rha_len in rha_len_range:
                # logging.info(f'PAM position: {pam_position}, PBS length: {pbs_len}, RHA length: {rha_len}')
                pbs_location_l.append(nicking_site - pbs_len)
                pbs_location_r.append(nicking_site)
                lha_location_l.append(nicking_site)
                lha_location_r.append(edit_position)
                rha_location_l.append(edit_position + edit_length)
                rha_location_r.append(edit_position + edit_length + rha_len)
                protospacer_location_l.append(pam_position - 20)
                protospacer_location_r.append(pam_position)
                wt_sequences.append(wt_sequence[protospacer_location_l[-1] - 10: protospacer_location_r[-1] + 89])
                mut_sequences.append(mut_sequence[protospacer_location_l[-1] - 10: protospacer_location_r[-1] + 89])
                # TODO figure out deepspcas9 score
                sp_cas9_score.append(0.5)
                rtt_location_l.append(lha_location_l[-1])
                rtt_location_r.append(rha_location_r[-1])
                mut_types.append(mut_type)

    
    df = pd.DataFrame({
        'pbs_location_l': pbs_location_l,
        'pbs_location_r': pbs_location_r,
        'lha_location_l': lha_location_l,
        'lha_location_r': lha_location_r,
        'rha_location_l': rha_location_l,
        'rha_location_r': rha_location_r,
        'protospacer_location_l': protospacer_location_l,
        'protospacer_location_r': protospacer_location_r,
        'rtt_location_l': rtt_location_l,
        'rtt_location_r': rtt_location_r,
        'sp_cas9_score': sp_cas9_score,
        'mut_type': mut_types,
        'wt_sequence': wt_sequences,
        'mut_sequence': mut_sequences
    })

    return df
    
def match_pam(sequence: str, pam: str) -> bool:
    """
    Check if the sequence is a valid PAM sequence

    Args:
        sequence (str): DNA sequence
        pam (str): PAM sequence
    
    Returns:
        bool: True if the sequence is a valid PAM sequence, False otherwise
    """
    # N refers to any nucleotide
    for i in range(len(pam)):
        if pam[i] != 'N' and sequence[i] != pam[i]:
            return False
    return True
    

def index(request):
    log.info('Index request received')
    return render(request, 'predictapp/index.html')