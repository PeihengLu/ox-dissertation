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
        
        # return the pegRNA design in std format
        pegRNAs = propose_pegrna(sequence, pam_table[pe], cellline in trained_on_pridict_only)
        
        # load all models trained on the specified cell line and prime editors
        # then takes an average of the predictions

        # parse user's input
        return JsonResponse({'prediction': prediction})
    else:
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
            mutation_type = 'Insertion'
            correction_length = len(edited_base)
        else:
            print(sequence)
            raise ValueError
    else:
        original_seq = five_prime_seq + original_base + three_prime_seq
        if edited_base == '-':
            mutation_type = 'Deletion'
            correction_length = len(original_base)
        elif len(original_base) == 1 and len(edited_base) == 1:
            if isDNA(original_base) and isDNA(edited_base):  # check if only AGCT is in bases
                mutation_type = '1bpReplacement'
                correction_length = len(original_base)
            else:
                print(sequence)
                print('Non DNA bases found in sequence! Please check your input sequence.')
                raise ValueError
        elif len(original_base) > 1 or len(edited_base) > 1:
            if isDNA(original_base) and isDNA(edited_base):  # check if only AGCT is in bases
                mutation_type = 'MultibpReplacement'
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

    if edited_base == '-':
        edited_seq = five_prime_seq + three_prime_seq
    else:
        edited_seq = five_prime_seq + edited_base.lower() + three_prime_seq

    if isDNA(edited_seq) and isDNA(original_seq):  # check whether sequences only contain AGCT
        pass
    else:
        raise ValueError

    basebefore_temp = five_prime_seq[
                      -1:]  # base before the edit, could be changed with baseafter_temp if Rv strand is targeted (therefore the "temp" attribute)
    baseafter_temp = three_prime_seq[:1]  # base after the edit

    editposition_left = len(five_prime_seq)
    editposition_right = len(three_prime_seq)
    return original_base, edited_base, original_seq, edited_seq, editposition_left, editposition_right, mutation_type, correction_length, basebefore_temp, baseafter_temp

def propose_pegrna(dna_sequence: str, pam: str, pridicrt_only: bool) -> str:
    pbs_len_range = range(8, 18) if not pridicrt_only else [13] 
    lha_len_range = range(0, 13)
    rha_len_range = range(7, 12)
    

def index(request):
    return render(request, 'predictapp/index.html')