from django.shortcuts import render

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
        
        # sequence = primesequenceparsing(sequence)
        
        trained_on_pridict_only = ['k562', 'adv']
        
        # return the pegRNA design in std format
        pegRNAs = propose_pegrna(sequence, pam_table[pe], cellline in trained_on_pridict_only, edit_length=1, pam=pam_table[pe], pridict_only=cellline in trained_on_pridict_only)
        
        # load all models trained on the specified cell line and prime editors
        # then takes an average of the predictions
        
        # # make dummy prediction
        # prediction = 

        # parse user's input
        return JsonResponse(pegRNAs)
    else:
        log.error('Invalid request method')
        return JsonResponse({'error': 'Invalid request method'}, status=400)
    
def isDNA(sequence: str) -> bool:
    """
    Function to check whether a given sequence only contains AGCT.

    Parameters
    ----------
    sequence : str
        Sequence to be checked.

    Returns
    -------
    bool
        True if only AGCT, False otherwise.

    """
    return set(sequence).issubset('AGCT')


def primesequenceparsing(sequence: str) -> object:
    """
    Function which takes target sequence with desired edit as input and 
    editing characteristics as output. Edit within brackets () and original
    equence separated by backslash from edited sequence: (A/G) == A to G mutation.
    Placeholder for deletions and insertions is '-'.

    Parameters
    ----------
    sequence : str
        Target sequence with desired edit in brackets ().

    Returns
    -------
    five_prime_seq: str
    three_prime_seq: str
    original_seq: str
    edited_seq: str
    original_base: str
    edited_base: str
    editposition: int
        DESCRIPTION.

    """
    
    sequence = sequence.replace('\n','')  # remove any spaces or linebreaks in input
    sequence = sequence.replace(' ','')
    sequence = sequence.upper()
    if sequence.count('(') != 1:
        print(sequence)
        print('More or less than one bracket found in sequence! Please check your input sequence.')
        raise ValueError

    five_prime_seq = sequence.split('(')[0]
    three_prime_seq = sequence.split(')')[1]

    sequence_set = set(sequence)
    if '/' in sequence_set:
        original_base = sequence.split('/')[0].split('(')[1]
        edited_base = sequence.split('/')[1].split(')')[0]

        # edit flanking bases should *not* be included in the brackets
        if (original_base[0] == edited_base[0]) or (original_base[-1] == edited_base[-1]):
            print(sequence)
            print('Flanking bases should not be included in brackets! Please check your input sequence.')
            raise ValueError
    elif '+' in sequence_set:  #insertion
        original_base = '-'
        edited_base = sequence.split('+')[1].split(')')[0]
    elif '-' in sequence_set:  #deletion
        original_base = sequence.split('-')[1].split(')')[0]
        edited_base = '-'

    # ignore "-" in final sequences (deletions or insertions)
    if original_base == '-':
        original_seq = five_prime_seq + three_prime_seq
        if edited_base != '-':
            mutation_type = 1
            correction_length = len(edited_base)
        else:
            print(sequence)
            raise ValueError
    else:
        original_seq = five_prime_seq + original_base + three_prime_seq
        if edited_base == '-':
            mutation_type = 2
            correction_length = len(original_base)
        elif len(original_base) == 1 and len(edited_base) == 1:
            if isDNA(original_base) and isDNA(edited_base):  # check if only AGCT is in bases
                mutation_type = 0
                correction_length = len(original_base)
            else:
                print(sequence)
                print('Non DNA bases found in sequence! Please check your input sequence.')
                raise ValueError
        elif len(original_base) > 1 or len(edited_base) > 1:
            if isDNA(original_base) and isDNA(edited_base):  # check if only AGCT is in bases
                mutation_type = 0
                if len(original_base) == len(
                        edited_base):  # only calculate correction length if replacement does not contain insertion/deletion
                    correction_length = len(original_base)
                else:
                    print(sequence)
                    print('Only 1bp replacements or replacements of equal length (before edit/after edit) are supported! Please check your input sequence.')
                    raise ValueError
            else:
                print(sequence)
                print('Non DNA bases found in sequence! Please check your input sequence.')
                raise ValueError

    if edited_base == '-': # deletion
        edited_seq = five_prime_seq + three_prime_seq
    else: 
        edited_seq = five_prime_seq + edited_base.lower() + three_prime_seq

    

    if isDNA(edited_seq) and isDNA(original_seq):  # check whether sequences only contain AGCT
        pass
    else:
        raise ValueError

    # basebefore_temp = five_prime_seq[
    #                   -1:]  # base before the edit, could be changed with baseafter_temp if Rv strand is targeted (therefore the "temp" attribute)
    # baseafter_temp = three_prime_seq[:1]  # base after the edit

    editposition = len(five_prime_seq)
    return original_base, edited_base, original_seq, edited_seq, editposition, mutation_type, correction_length#, basebefore_temp, baseafter_temp

def propose_pegrna(wt_sequence: str, edit_position: int, mut_type: int, edit_length: int, pam: str, pridict_only: bool) -> pd.DataFrame:
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