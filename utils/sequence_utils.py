'''
Functions for preprocessing sequence data
'''
import numpy as np
import pandas as pd
import torch

# onehot encode a sequence
# produce a 4 x n numpy array
def onehot_encode(seq: str) -> np.ndarray:
    seq = seq.upper()
    encoding = np.zeros((4, len(seq)), dtype=np.float32)
    for i, base in enumerate(seq):
        if base == 'A':
            encoding[0, i] = 1
        elif base == 'C':
            encoding[1, i] = 1
        elif base == 'G':
            encoding[2, i] = 1
        elif base == 'T':
            encoding[3, i] = 1
        else:
            continue
    return encoding
# onehot encode a pandas series of sequences
# produce a 4 x n x m numpy array
def onehot_encode_series(series: pd.Series) -> torch.Tensor:
    n = len(series)
    m = len(series.iloc[0])
    encoding = np.zeros((n, 4, m), dtype=np.float32)
    for i, seq in enumerate(series):
        encoding[i] = onehot_encode(seq)
    return torch.tensor(encoding, dtype=torch.float32)


def get_compliment_dna_to_rna(sequence: str) -> str:
    '''
    Returns the RNA compliment of a given DNA sequence
    '''
    sequence = sequence.upper()
    rna_compliment = ''
    for base in sequence:
        if base == 'A':
            rna_compliment += 'U'
        elif base == 'C':
            rna_compliment += 'G'
        elif base == 'G':
            rna_compliment += 'C'
        elif base == 'T':
            rna_compliment += 'A'
        else:
            continue
    return rna_compliment

def get_compliment_rna_to_dna(sequence: str) -> str:
    '''
    Returns the DNA compliment of a given RNA sequence
    '''
    sequence = sequence.upper()
    dna_compliment = ''
    for base in sequence:
        if base == 'A':
            dna_compliment += 'T'
        elif base == 'C':
            dna_compliment += 'G'
        elif base == 'G':
            dna_compliment += 'C'
        elif base == 'U':
            dna_compliment += 'A'
        else:
            continue
    return dna_compliment

def get_compliment_dna_to_dna(sequence: str) -> str:
    '''
    Returns the DNA compliment of a given DNA sequence
    '''
    sequence = sequence.upper()
    dna_compliment = ''
    for base in sequence:
        if base == 'A':
            dna_compliment += 'T'
        elif base == 'C':
            dna_compliment += 'G'
        elif base == 'G':
            dna_compliment += 'C'
        elif base == 'T':
            dna_compliment += 'A'
        else:
            continue
    return dna_compliment