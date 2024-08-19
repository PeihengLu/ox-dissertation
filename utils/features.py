'''
Extracting the features from specific sequences
'''

import Bio
import Bio.SeqUtils
import Bio.SeqUtils.MeltingTemp
import ViennaRNA

from typing import List, Tuple

# test the functions
def test():
    '''
    Test the functions in the module
    '''
    # example sequence 100bp long
    sequence = 'CTGAATATTGACATACATGAAGTTCGTATTGGCTGATCACTAACTACAAGTGGAATGTTACTTGGTAAAGCTTAAAAAAATTCTAGTTGGTTTAACGCG'
    print(get_gc_content_and_count(sequence))
    print(get_melting_temperature(sequence))
    print(get_minimum_free_energy(sequence))
    print(get_consecutive_n_sequences('A', sequence))
    print(get_consecutive_n_sequences('T', sequence))

def get_melting_temperature(sequence: str, table: str, c_seq: str = None) -> float:
    '''
    Returns the melting temperature of a given sequence
    '''
    # remove the N in the sequence
    if 'N' in sequence:
        sequence = sequence.replace('N', '')
        
    if len(sequence) == 0:
        return 0
    
    tables = {'R_DNA_NN1': Bio.SeqUtils.MeltingTemp.R_DNA_NN1, 'DNA_NN3': Bio.SeqUtils.MeltingTemp.DNA_NN3}
    
    if c_seq:
        return Bio.SeqUtils.MeltingTemp.Tm_NN(sequence, nn_table=tables[table], c_seq=c_seq)
    return Bio.SeqUtils.MeltingTemp.Tm_NN(sequence, nn_table=tables[table])

def get_minimum_free_energy(sequence: str) -> float:
    '''
    Returns the minimum free energy of a given sequence
    '''        
    if 'N' in sequence:
        sequence = sequence.replace('N', '')
        
    if len(sequence) == 0:
        return 0

    # fold the sequence
    fc = ViennaRNA.fold_compound(sequence)
    ss, mfe = fc.mfe()
    return mfe

def get_gc_content_and_count(sequence: str) -> Tuple[int, int]:
    '''
    Returns the GC content and count of a given sequence
    '''   
    if 'N' in sequence:
        sequence = sequence.replace('N', '')
        
    if len(sequence) == 0:
        return 0, 0

    gc_count = 0
    for base in sequence:
        if base in ['G', 'C']:
            gc_count += 1
    
    gc_content = gc_count / len(sequence)
    
    return gc_content, gc_count

def get_consecutive_n_sequences(n: str, sequence: str) -> List[str]: 
    '''
    Returns the sequences of poly-n
    '''
    if 'N' in sequence:
        # remove the Ns
        sequence = sequence.replace('N', '')
        
    poly_n_sequences = []

    i = 0
    while i < len(sequence):
        if sequence[i] != n:
            i += 1
            continue

        length = 0
        while i + length < len(sequence) and sequence[i + length] == n:
            length += 1
        if length > 1:
            poly_n_sequences.append(sequence[i:i+length])
        i += length     
    
    return poly_n_sequences

if __name__ == '__main__':
    test()