from django.shortcuts import render
from typing import Tuple

# Create your views here.
import json
import torch
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import pandas as pd
import numpy as np
import sys
# add the absolute path of the project to the sys.path
sys.path.append('/home/peiheng/development/ox-dissertation')

import logging
log = logging.getLogger(__name__)
from .deepcas9 import runprediction
from models.ensemble_bagging import predict_df
from utils.data_utils import convert_to_ensemble_df, match_pam

# Define your PyTorch model (replace with your actual model)
class DummyModel(torch.nn.Module):
    def forward(self, x):
        return torch.tensor([x.sum()])

model = DummyModel()

def deepcas9(deepcas9seqlist):
    """Perform DeepCas9 prediction on 30bp stretches of protospacer + targetseq for each protospacer."""
    deepcas9scorelist = runprediction(deepcas9seqlist)
    print('deepcas9 calculating...')
    deepcas9scorelist = [round(x, 2) for x in deepcas9scorelist]
    return deepcas9scorelist

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
            'nrch_pe4max': 'NRCH',
            'pe2': 'NGG',
            'nrch_pe2': 'NRCH',
            'nrch_pe2max': 'NRCH',
        }
        
        trained_on_pridict_only = ['k562', 'adv']

        wt_sequence, mut_sequence, edit_position, mut_type, edit_length = prime_sequence_parsing(sequence)

        # return the pegRNA design in std format
        pegRNAs = propose_pegrna(wt_sequence=wt_sequence, mut_sequence=mut_sequence, edit_position=edit_position, mut_type=mut_type, edit_length=edit_length, pam=pam_table[pe], pridict_only=cellline in trained_on_pridict_only)

        pegRNAs_aligned = pegRNAs.copy()
        pegRNAs_aligned['pbs-location-l'] = pegRNAs['pbs-location-l'] - pegRNAs['protospacer-location-l'] + 10
        pegRNAs_aligned['pbs-location-r'] = pegRNAs['pbs-location-r'] - pegRNAs['protospacer-location-l'] + 10
        pegRNAs_aligned['lha-location-l'] = pegRNAs['lha-location-l'] - pegRNAs['protospacer-location-l'] + 10
        pegRNAs_aligned['lha-location-r'] = pegRNAs['lha-location-r'] - pegRNAs['protospacer-location-l'] + 10
        pegRNAs_aligned['rha-location-l'] = pegRNAs['rha-location-l'] - pegRNAs['protospacer-location-l'] + 10
        pegRNAs_aligned['rha-location-r'] = pegRNAs['rha-location-r'] - pegRNAs['protospacer-location-l'] + 10
        pegRNAs_aligned['rtt-location-l'] = pegRNAs['rtt-location-l'] - pegRNAs['protospacer-location-l'] + 10
        pegRNAs_aligned['rtt-location-r'] = pegRNAs['rtt-location-r'] - pegRNAs['protospacer-location-l'] + 10
        pegRNAs_aligned['protospacer-location-l'] = 10
        pegRNAs_aligned['protospacer-location-r'] = 30

        # log.info(f'Proposed pegRNAs: {pegRNAs.columns}')
        ensemble_data = convert_to_ensemble_df(pegRNAs_aligned, pam=pam_table[pe])
        # log.info(f'Running ensemble prediction on {len(ensemble_data)} sequences')
        # log.info(f'PE: {pe}, Cellline: {cellline}')
        # log.info(f'Ensemble data: {ensemble_data}') 
        efficiencies = predict_df(ensemble_data, cell_line=cellline, pe=pe)
        
        # load all models trained on the specified cell line and prime editors
        # then takes an average of the predictions
        pegRNAs['predicted_efficiency'] = efficiencies
        # change all - in the column names to _
        pegRNAs.columns = [col.replace('-', '_') for col in pegRNAs.columns]

        # order by predicted efficiency
        pegRNAs = pegRNAs.sort_values(by='predicted_efficiency', ascending=False)

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
    pbs_len_range = np.arange(8, 16) if not pridict_only else [13] 
    lha_len_range = np.arange(0, 13)
    rha_len_range = np.arange(7, 12)
    
    # in the range of lha length, scan for PAM sequences
    # edit must start before 3bp upstream of the PAM
    edit_to_pam_range = lha_len_range + 3
    
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
    edit_lengths = []
    
    for pam_distance_to_edit in edit_to_pam_range:
        # no valid PAM sequence
        # PAM is 3bp downstream of nicking site
        # nicking site is the end of PBS and start of LHA
        pam_position = edit_position - pam_distance_to_edit
        if not match_pam(wt_sequence[pam_position: pam_position + len(pam)] , pam):
            continue
        # check if the protospacer start with a G
        # if wt_sequence[pam_position - 20] != 'G':
        #     continue
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
                rtt_location_l.append(lha_location_l[-1])
                rtt_location_r.append(rha_location_r[-1])
                mut_types.append(mut_type)
                edit_lengths.append(edit_length)

    if len(protospacer_location_l) == 0:
        log.info('No valid pegRNA found')
        # keep searching until the end of the sequence
        edit_to_pam_range = np.arange(edit_to_pam_range[-1] + 1, 70)
        for pam_distance_to_edit in edit_to_pam_range:
            # no valid PAM sequence
            # PAM is 3bp downstream of nicking site
            # nicking site is the end of PBS and start of LHA
            pam_position = edit_position - pam_distance_to_edit
            if not match_pam(wt_sequence[pam_position: pam_position + len(pam)] , pam):
                continue
            # check if the protospacer start with a G
            if wt_sequence[pam_position - 20] != 'G':
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
                    rtt_location_l.append(lha_location_l[-1])
                    rtt_location_r.append(rha_location_r[-1])
                    mut_types.append(mut_type)
                    edit_lengths.append(edit_length)
            break

    spcas9_sequence_list = []
    # spcas9 takes 30 bp long sequence starting from 4bp upstream of the protospacer
    for i in range(len(protospacer_location_l)):
        spcas9_sequence_list.append(wt_sequences[i][protospacer_location_l[i] - 4: protospacer_location_l[i] + 26])
    log.info(f'Running DeepCas9 prediction on {len(spcas9_sequence_list)} sequences')
    spcas9_score = runprediction(spcas9_sequence_list)
    log.info(f'DeepCas9 prediction complete')
    log.info(f'returns {len(sp_cas9_score)} scores')

    
    df = pd.DataFrame({
        'pbs-location-l': pbs_location_l,
        'pbs-location-r': pbs_location_r,
        'lha-location-l': lha_location_l,
        'lha-location-r': lha_location_r,
        'rha-location-l': rha_location_l,
        'rha-location-r': rha_location_r,
        'protospacer-location-l': protospacer_location_l,
        'protospacer-location-r': protospacer_location_r,
        'rtt-location-l': rtt_location_l,
        'rtt-location-r': rtt_location_r,
        'spcas9-score': spcas9_score,
        'mut-type': mut_types,
        'wt-sequence': wt_sequences,
        'mut-sequence': mut_sequences,
        'edit-len': edit_lengths,
    })

    return df
    
def index(request):
    log.info('Index request received')
    return render(request, 'predictapp/index.html')