'''
Utilities for handling and converting standard data format to and from various format
As well as data pre processing
'''
import pandas as pd
import numpy as np
from os.path import join as pjoin, basename, isfile
import ast
import tqdm
import torch
from typing import List, Tuple

from utils.features import get_gc_content_and_count, get_melting_temperature, get_minimum_free_energy, get_consecutive_n_sequences

# =============================================================================
# Data format conversion
# =============================================================================

def convert_from_deepprime_org(data: pd.DataFrame, cell_line: str, editor: str, source: str = None) -> None:
    '''
    Convert the data from DeepPrime format to the standard format
    '''
    if not source:
        target = f"std-dp-{cell_line}-{editor}.csv"
    else:
        target = f"std-dp_{source}-{cell_line}-{editor}.csv"
        
    # if isfile(pjoin('../', 'std', target)):
    #     return

    # replace the '-' in editor and cell line with '_'
    cell_line = cell_line.lower()
    editor = editor.lower()
    cell_line = cell_line.replace('-', '_')
    editor = editor.replace('-', '_')

    output = []

    # result columns
    result_columns = ['cell-line', 'group-id', 'mut-type', 'edit-len', 'wt-sequence', 'mut-sequence', 'protospacer-location-l', 'protospacer-location-r', 'pbs-location-l', 'pbs-location-r', 'rtt-location-l', 'rtt-location-r', 'lha-location-l', 'lha-location-r', 'rha-location-l', 'rha-location-r', 'spcas9-score', 'editing-efficiency']

    g_id = 0
    prev = ""

    # iterate over the data
    for ind, item in tqdm.tqdm(data.iterrows(), total=len(data)):
        try:
            wt_sequence = item["Wide target sequence (Target 74bps = 4bp neighboring sequence + 20 bp protospacer + 3 bp NGG + 47 bp neighboring sequence)"]
        except:
            wt_sequence = item["Wide target sequence\n(Target 74bps = 4bp neighboring sequence + 20 bp protospacer + 3 bp NGG + 47 bp neighboring sequence)"]
        try:
            edited_sequence = item["Edited target sequence (Target 74bps = RT-PBS corresponding region and masked by 'x')"]
        except:
            edited_sequence = item["Edited target sequence\n(Target 74bps = RT-PBS corresponding region and masked by 'x')"]

        group_id = ind
        edit_len = item['Edit_len']
        rha_len = item['RHA_len']
        pbs_rtt_len = item['RT-PBSlen']

        # edit type
        if item['type_sub']:
            mut_type = 0
        elif item['type_ins']:
            mut_type = 1
        elif item['type_del']:
            mut_type = 2
        
        protospacer_location_l = 4
        protospacer_location_r = 24

        protospacer = wt_sequence[protospacer_location_l:protospacer_location_r]
        
        # assign group based on target loci
        if prev:
            if protospacer == prev:
                group_id = g_id
            else:
                g_id += 1
                group_id = g_id
        else:
            group_id = g_id
        prev = protospacer

        pbs_len = item['PBSlen']
        rtt_len = item['RTlen']

        pbs_location_l = -1
        for ind, c in enumerate(edited_sequence):
            if c != 'x':
                pbs_location_l = ind
                break
        pbs_location_r = pbs_location_l + pbs_len
        lha_location_l = pbs_location_r
        if mut_type == 2: # deletion
            lha_location_r = pbs_location_r + (pbs_rtt_len - pbs_len - rha_len)
        else:
            lha_location_r = pbs_location_r + (pbs_rtt_len - pbs_len - rha_len - edit_len)

        if mut_type == 2: # deletion
            rha_location_wt_l = pbs_location_l + pbs_rtt_len - rha_len + edit_len
            rha_location_wt_r = pbs_location_l + pbs_rtt_len + edit_len
            rha_location_mut_l = pbs_location_l + pbs_rtt_len - rha_len
            rha_location_mut_r = pbs_location_l + pbs_rtt_len
        elif mut_type == 1: # insertion
            rha_location_wt_l = pbs_location_l + pbs_rtt_len - rha_len - edit_len
            rha_location_wt_r = pbs_location_l + pbs_rtt_len - edit_len
            rha_location_mut_l = pbs_location_l + pbs_rtt_len - rha_len
            rha_location_mut_r = pbs_location_l + pbs_rtt_len
        else: # length does not change
            rha_location_wt_l = pbs_location_l + pbs_rtt_len - rha_len
            rha_location_wt_r = pbs_location_l + pbs_rtt_len
            rha_location_mut_l = pbs_location_l + pbs_rtt_len - rha_len
            rha_location_mut_r = pbs_location_l + pbs_rtt_len 

        rtt_location_wt_l = pbs_location_r
        rtt_location_wt_r = rha_location_wt_r
        rtt_location_mut_l = pbs_location_r
        rtt_location_mut_r = rha_location_mut_r
        
        rtt_location_l = rtt_location_wt_l
        if mut_type == 2: # deletion, mut sequence is padded with N
            rtt_location_r = rtt_location_wt_r
        else:
            rtt_location_r = rtt_location_mut_r
            
        rha_location_l = rha_location_wt_l
        if mut_type == 2: # deletion, mut sequence is padded with N
            rha_location_r = rha_location_wt_r
        else:
            rha_location_r = rha_location_mut_r

        # remove the mask of the mutated sequence
        mut_sequence = ''
        mut_sequence += wt_sequence[:lha_location_r]
        mut_sequence += edited_sequence[lha_location_r:rha_location_mut_r]
        if mut_type == 1: # insertion
            mut_sequence += wt_sequence[rha_location_wt_r:len(wt_sequence)-edit_len]
        else:
            mut_sequence += wt_sequence[rha_location_wt_r:]
            
        wt_sequence, mut_sequence = align_wt_mut_sequences(wt_sequence, mut_sequence, lha_location_r, edit_length=edit_len, edit_type=mut_type)
        
        spcas9_score = item['DeepSpCas9_score']
        editing_efficiency = item['Measured_PE_efficiency']

        # # pad the mutated sequence to the same length as the wildtype sequence
        # if len(mut_sequence) < len(wt_sequence):
        #     mut_sequence += 'N' * (len(wt_sequence) - len(mut_sequence))
        
        output.append([cell_line, group_id, mut_type, edit_len, wt_sequence, mut_sequence, protospacer_location_l, protospacer_location_r, pbs_location_l, pbs_location_r, rtt_location_l, rtt_location_r, lha_location_l, lha_location_r, rha_location_l, rha_location_r, spcas9_score, editing_efficiency])

    # save the extracted information
    output_df = pd.DataFrame(output, columns=result_columns, index=None)
    # add fold column
    output_df = k_fold_cross_validation_split(output_df, 5)
    output_df.to_csv(pjoin('..', 'std', target), index=False)

def convert_to_deepprime(source: str) -> None:
    '''
    Convert the data from the standard format to the DeepPrime format usable for training
    '''
    # originating source
    _, org, cell_line, editor = basename(source).split('.')[0].split('-') 
    
    target = f"dp-{org}-{cell_line}-{editor}.csv"

    if isfile(pjoin('..', 'deepprime', target)):
        return
    
    # if basename(source) != 'std-dp-a549-pe2max.csv':
    #     return
    
    # load the data
    data = pd.read_csv(source)

    output = []

    columns = ['wt-sequence', 'mut-sequence'] + [s+'-length' for s in ['pbs', 'rt', 'extension', 'edit', 'rha']] + ['edit-position'] + ['edit-type-' + s for s in ['replacement', 'insertion', 'deletion']] + [f"{s}-melting-temperature" for s in ['pbs', 'rtt-wt-cdna', 'rtt-wt-cdna-new', 'rtt-cdna', 'rtt', 'delta']] + [f"{s}-gc-content" for s in ['pbs', 'rtt', 'extension']] + [f"{s}-gc-count" for s in ['pbs', 'rtt', 'extension']] + [f"{s}-minimum-free-energy" for s in ['extension', 'spacer']] + ['spcas9-score', 'group-id', 'editing-efficiency', 'fold']

    for ind, item in tqdm.tqdm(data.iterrows(), total=len(data)):
        wt_sequence = item['wt-sequence']
        mut_sequence = item['mut-sequence']

        pbs = wt_sequence[item['pbs-location-l']:item['pbs-location-r']]
        pbs = get_compliment_dna_to_dna(pbs)
        pbs = get_compliment_dna_to_rna(pbs)
        rtt_wt_cdna = wt_sequence[item['rtt-location-wt-l']:item['rtt-location-wt-r']]
        rtt_wt_cdna = get_compliment_dna_to_dna(rtt_wt_cdna)
        rtt_wt_cdna_new = wt_sequence[item['rtt-location-mut-l']:item['rtt-location-mut-r']]
        rtt_wt_cdna_new = get_compliment_dna_to_dna(rtt_wt_cdna_new)
        rtt_cdna = mut_sequence[item['rtt-location-mut-l']:item['rtt-location-mut-r']]
        rtt_cdna = get_compliment_dna_to_dna(rtt_cdna)
        rtt = mut_sequence[item['rtt-location-mut-l']:item['rtt-location-mut-r']]
        rtt = get_compliment_dna_to_dna(rtt)
        rtt = get_compliment_dna_to_rna(rtt)
        extension = mut_sequence[item['pbs-location-l']:item['rtt-location-mut-r']]
        extension = get_compliment_dna_to_dna(extension)
        extension = get_compliment_dna_to_rna(extension)
        rha = wt_sequence[item['rha-location-wt-l']:item['rha-location-wt-r']]
        rha = get_compliment_dna_to_dna(rha)
        rha = get_compliment_dna_to_rna(rha)
        spacer = wt_sequence[item['protospacer-location-l']:item['protospacer-location-r']]
        spacer = get_compliment_dna_to_rna(spacer)

        pre_protospacer_length = 4
        post_protospacer_length = 50

        # crop to the required length
        if len(wt_sequence) != 20 + pre_protospacer_length + post_protospacer_length:
            if pre_protospacer_length - item['protospacer-location-l'] > 0:
                # pad with N if the sequence is too short on the left
                for i in range(pre_protospacer_length-item['protospacer-location-l']):
                    wt_sequence = 'N' + wt_sequence
                    mut_sequence = 'N' + mut_sequence
                edit_position = item['lha-location-r'] + pre_protospacer_length - item['protospacer-location-l']
            else:
                wt_sequence = wt_sequence[item['protospacer-location-l'] - pre_protospacer_length:]
                mut_sequence = mut_sequence[item['protospacer-location-l'] - pre_protospacer_length:]
                edit_position = item['lha-location-r'] + item['protospacer-location-l'] - pre_protospacer_length
            if len(wt_sequence) < 20 + pre_protospacer_length + post_protospacer_length:
                # pad with N if the sequence is too short on the right
                for i in range(20 + pre_protospacer_length + post_protospacer_length - len(wt_sequence)):
                    wt_sequence += 'N'
                    mut_sequence += 'N'
            else:
                wt_sequence = wt_sequence[:20 + pre_protospacer_length + post_protospacer_length]
                mut_sequence = mut_sequence[:20 + pre_protospacer_length + post_protospacer_length]
        else:
            # edit position
            edit_position = item['lha-location-r']

        # edit type
        edit_type = [0, 0, 0]
        edit_type[item['mut-type']] = 1
        

        # melting temperature
        pbs_melting_temperature = get_melting_temperature(pbs, 'R_DNA_NN1')
        rtt_wt_cdna_melting_temperature = get_melting_temperature(rtt_wt_cdna, 'DNA_NN3')
        rtt_wt_cdna_new_melting_temperature = get_melting_temperature(rtt_wt_cdna_new, 'DNA_NN3')
        rtt_cdna_melting_temperature = get_melting_temperature(rtt_cdna, 'DNA_NN3')
        rtt_melting_temperature = get_melting_temperature(rtt, 'R_DNA_NN1')
        delta_melting_temperature = rtt_cdna_melting_temperature - rtt_wt_cdna_melting_temperature

        # gc content and count
        pbs_gc_content, pbs_gc_count = get_gc_content_and_count(pbs)
        rtt_gc_content, rtt_gc_count = get_gc_content_and_count(rtt)
        extension_gc_content, extension_gc_count = get_gc_content_and_count(extension)

        # minimum free energy
        extension_minimum_free_energy = get_minimum_free_energy(extension + 'TTTTTT')
        spacer_minimum_free_energy = get_minimum_free_energy(spacer)

        spcas9_score = item['spcas9-score']
        editing_efficiency = item['editing-efficiency']
        
        edit_len = item['edit-len']
        
        # mask the mut-sequence by turning all non pbs and rtt region to N
        mut_sequence = 'N' * item['pbs-location-l'] + mut_sequence[item['pbs-location-l']:item['rtt-location-mut-r']] + 'N' * (len(mut_sequence) - item['rtt-location-mut-r'])

        output.append([wt_sequence, mut_sequence, len(pbs), len(rtt), len(extension), edit_len, len(rha), edit_position] + edit_type + [pbs_melting_temperature, rtt_wt_cdna_melting_temperature, rtt_wt_cdna_new_melting_temperature, rtt_cdna_melting_temperature, rtt_melting_temperature, delta_melting_temperature] + [pbs_gc_content, rtt_gc_content, extension_gc_content] + [pbs_gc_count, rtt_gc_count, extension_gc_count] + [extension_minimum_free_energy, spacer_minimum_free_energy] + [spcas9_score, item['group-id'], editing_efficiency, item['fold']])

    # save the extracted information
    output_df = pd.DataFrame(output, columns=columns)
    # all numerical columns should be float32
    for col in columns:
        if col not in ['wt-sequence', 'mut-sequence']:
            output_df[col] = output_df[col].astype(np.float32)
    output_df.to_csv(pjoin('..', 'deepprime', target), index=False)
    

def convert_from_pridict2_org(data: pd.DataFrame) -> None:
    '''
    Convert the data from PRIDICT2 format to the standard format
    '''
    output = []
    
    # drop the rows where wt-sequence or mut-sequence is empty
    data = data.dropna(subset=['wide_initial_target', 'wide_mutated_target'])

    # cell lines
    cell_lines = {'HEKaverageedited': 'hek293t','K562averageedited': 'k562','K562MLH1dnaverageedited': 'k562mlh1dn','AdVaverageedited': 'adv'}

    # result columns
    result_columns = ['cell-line', 'group-id', 'mut-type', 'edit-len', 'wt-sequence', 'mut-sequence', 'protospacer-location-l', 'protospacer-location-r', 'pbs-location-l', 'pbs-location-r', 'rtt-location-l', 'rtt-location-r', 'lha-location-l', 'lha-location-r', 'rha-location-l', 'rha-location-r', 'spcas9-score', 'editing-efficiency']

    # enum of mutation types
    mutation_types = ['1bpReplacement', 'MultibpReplacement', 'Insertion', 'Deletion']
    
    group_prev = -1
    group_id = -1

    # extract the important information
    for ind, item in tqdm.tqdm(data.iterrows(), total=len(data)):
        if item['group'] != group_prev:
            group_id += 1
            group_prev = item['group']
        wt_sequence = item['wide_initial_target']
        mut_sequence = item['wide_mutated_target']
        
        protospacer_location = ast.literal_eval(item['protospacerlocation_only_initial'])
        pbs_location = ast.literal_eval(item['PBSlocation'])
        rtt_location_wt = ast.literal_eval(item['RT_initial_location'])
        rtt_location_mut = ast.literal_eval(item['RT_mutated_location'])

        protospacer_location_l = protospacer_location[0]
        protospacer_location_r = protospacer_location[1] + 1
        pbs_location_l = pbs_location[0] + 1
        pbs_location_r = pbs_location[1] + 1
        rtt_location_wt_l = rtt_location_wt[0] + 1
        rtt_location_wt_r = rtt_location_wt[1] + 1
        rtt_location_mut_l = rtt_location_mut[0] + 1
        rtt_location_mut_r = rtt_location_mut[1] + 1

        mut_type = mutation_types.index(item['Mutation_Type'])
        if mut_type == 3: # deletion
            mut_type = 2
        elif mut_type == 2: # insertion
            mut_type = 1
        else: # replacement
            mut_type = 0

        rha_length = len(item['RTToverhang'])
        edit_length = int(item['Correction_Length'])

        if mut_type != 2: # not deletion
            lha_length = rtt_location_mut_r - rtt_location_mut_l - rha_length - edit_length
            lha_location_l = rtt_location_wt_l
            lha_location_r = rtt_location_wt_l + lha_length
            rha_location_wt_l = rtt_location_wt_r - rha_length
            rha_location_wt_r = rtt_location_wt_r
            rha_location_mut_l = rtt_location_mut_r - rha_length
            rha_location_mut_r = rtt_location_mut_r
        else:
            lha_length = rtt_location_mut_r - rtt_location_mut_l - rha_length
            lha_location_l = rtt_location_wt_l
            lha_location_r = rtt_location_wt_l + lha_length
            rha_location_wt_l = rtt_location_wt_r - rha_length
            rha_location_wt_r = rtt_location_wt_r
            rha_location_mut_l = rtt_location_mut_r - rha_length
            rha_location_mut_r = rtt_location_mut_r
        spcas9_score = float(item['deepcas9'])
        
        wt_sequence, mut_sequence = align_wt_mut_sequences(wt_sequence, mut_sequence, lha_location_r, edit_length=edit_length, edit_type=mut_type)
        
        rtt_location_l = rtt_location_wt_l
        if mut_type == 2: # deletion, mut sequence is padded with N
            rtt_location_r = rtt_location_wt_r
        else:
            rtt_location_r = rtt_location_mut_r
            
        rha_location_l = rha_location_wt_l
        if mut_type == 2: # deletion, mut sequence is padded with N
            rha_location_r = rha_location_wt_r
        else:
            rha_location_r = rha_location_mut_r

        item_nan = item.isna()

        for cell_line in cell_lines:
            if not item_nan[cell_line]:
                output.append([cell_lines[cell_line], group_id, mut_type, edit_length, wt_sequence, mut_sequence, protospacer_location_l, protospacer_location_r, pbs_location_l, pbs_location_r, rtt_location_l, rtt_location_r, lha_location_l, lha_location_r, rha_location_l, rha_location_r, spcas9_score, item[cell_line]])


    # save the extracted information
    output_df = pd.DataFrame(output, columns=result_columns)
    # each cell line needs to be saved separately
    for cell_line in cell_lines.values():
        target = f"std-pd-{cell_line}-pe2.csv"
        cell_line_data = output_df[output_df['cell-line'] == cell_line]
        # add fold column
        cell_line_data = k_fold_cross_validation_split(cell_line_data, 5)
        cell_line_data.to_csv(pjoin('..', 'std', target), index=False)
        
# convert to pridict
def convert_to_pridict(source: str) -> None:
    """Convert the data from the standard format to the PRIDICT format

    Args:
        source (str): the complete path to the source file
    """
    # originating source
    _, org, cell_line, editor = basename(source).split('.')[0].split('-') 
    
    target = f"pd-{org}-{cell_line}-{editor}.csv"
    
    # if isfile(pjoin('..', 'pridict', target)):
    #     return
    
    # load the data
    data = pd.read_csv(source)
    
    columns = ['wt-sequence', 'mut-sequence'] + [s+'-length' for s in ['edit', 'pbs', 'rtt', 'rha']] + ['mut-type'] + [f"{s}-sequence-zero-length" for s in ['edit', 'preedit']] + ['protospacer-location'] + [f"{s}-location-l-relative-protospacer" for s in ['pbs', 'rtt', 'rha']] + [f"{s}-melting-temperature" for s in ['edit', 'extension', 'preedit', 'protospacer', 'rha', 'pbs']] + [f"{s}-minimum-free-energy" for s in ['extension', 'extension-scaffold', 'pbs', 'spacer', 'spacer-extension-scaffold', 'spacer-scaffold', 'rtt']] + ['group-id', 'editing-efficiency', 'fold']
    
    scaffold = 'GTTTTAGAGCTAGAAATAGCAAGTTAAAATAAGGCTAGTCCGTTATCAACTTGAAAAAGTGGCACCGAGTCGGTGC'
    # scaffold = 'G'
    scaffold = get_compliment_dna_to_rna(scaffold)
    
    output = []
    
    for ind, item in tqdm.tqdm(data.iterrows(), total=len(data)):
        wt_sequence = item['wt-sequence']
        mut_sequence = item['mut-sequence']
        
        rha = wt_sequence[item['rha-location-wt-l']:item['rha-location-wt-r']]
        rha = get_compliment_dna_to_dna(rha)
        rha = get_compliment_dna_to_rna(rha)
        pbs = wt_sequence[item['pbs-location-l']:item['pbs-location-r']]
        pbs = get_compliment_dna_to_dna(pbs)
        pbs = get_compliment_dna_to_rna(pbs)
        rtt = mut_sequence[item['rtt-location-mut-l']:item['rtt-location-mut-r']]
        rtt = get_compliment_dna_to_dna(rtt)
        rtt = get_compliment_dna_to_rna(rtt)
        extension = mut_sequence[item['pbs-location-l']:item['rtt-location-mut-r']]
        extension = get_compliment_dna_to_dna(extension)
        extension = get_compliment_dna_to_rna(extension)
        spacer = wt_sequence[item['protospacer-location-l']:item['protospacer-location-r']]
        spacer = get_compliment_dna_to_rna(spacer)
        
        mut_type = item['mut-type']
        edit_len = item['edit-len']
        
        edit_melting_temperature = get_melting_temperature(mut_sequence[item['lha-location-r']:item['rha-location-mut-l']], 'DNA_NN3')
        edit_sequence_zero_length = item['rha-location-mut-l'] - item['lha-location-r'] == 0
        
        preedit_melting_temperature = get_melting_temperature(wt_sequence[item['lha-location-r']:item['rha-location-wt-l']], 'DNA_NN3')
        preedit_sequence_zero_length = item['rha-location-wt-l'] - item['lha-location-r'] == 0
        
        pbs_melting_temperature = get_melting_temperature(pbs, 'R_DNA_NN1')
        rha_melting_temperature = get_melting_temperature(rha, 'R_DNA_NN1')
        extension_melting_temperature = get_melting_temperature(extension, 'R_DNA_NN1')
        protosacer_melting_temperature = get_melting_temperature(spacer, 'R_DNA_NN1')
        
        pbs_location_l = item['pbs-location-l'] - item['protospacer-location-l']
        rtt_location_l = item['rtt-location-mut-l'] - item['protospacer-location-l']
        rha_location_l = item['rha-location-mut-l'] - item['protospacer-location-l']
        protospacer_location_l = item['protospacer-location-l']
        
        extension_minimum_free_energy = get_minimum_free_energy(extension)
        extension_scaffold_minimum_free_energy = get_minimum_free_energy(extension + scaffold)
        pbs_minimum_free_energy = get_minimum_free_energy(pbs)
        spacer_minimum_free_energy = get_minimum_free_energy(spacer)
        spacer_extension_scaffold_minimum_free_energy = get_minimum_free_energy(spacer + extension + scaffold)
        spacer_scaffold_minimum_free_energy = get_minimum_free_energy(spacer + scaffold)
        rtt_minimum_free_energy = get_minimum_free_energy(rtt)
        
        
        output.append([wt_sequence, mut_sequence, edit_len, len(pbs), len(rtt), len(rha), mut_type, edit_sequence_zero_length, preedit_sequence_zero_length, protospacer_location_l, pbs_location_l, rtt_location_l, rha_location_l, edit_melting_temperature, extension_melting_temperature, preedit_melting_temperature, protosacer_melting_temperature, pbs_melting_temperature, rha_melting_temperature, extension_minimum_free_energy, extension_scaffold_minimum_free_energy, pbs_minimum_free_energy, spacer_minimum_free_energy, spacer_extension_scaffold_minimum_free_energy, spacer_scaffold_minimum_free_energy, rtt_minimum_free_energy, item['group-id'], item['editing-efficiency'], item['fold']])
    
    # save the extracted information
    output_df = pd.DataFrame(output, columns=columns)
    # all numerical columns should be float32
    for col in columns:
        if col not in ['wt-sequence', 'mut-sequence']:
            output_df[col] = output_df[col].astype(np.float32)
    output_df.to_csv(pjoin('..', 'pridict', target), index=False)
    
    
# convert to pridict
def convert_to_pridict_direct(data: pd.DataFrame) -> None:
    """Convert the data from the pridict format directly to the PRIDICT format

    Args:
        source (str): the complete path to the source file
    """
    # cell lines
    cell_lines = {'HEKaverageedited': 'hek293t','K562averageedited': 'k562','K562MLH1dnaverageedited': 'k562mlh1dn','AdVaverageedited': 'adv'}
    
    columns = ['wt-sequence', 'mut-sequence'] + [s+'-length' for s in ['edit', 'pbs', 'rtt', 'rha']] + ['mut-type'] + [f"{s}-sequence-zero-length" for s in ['edit', 'preedit']] + ['protospacer-location'] + [f"{s}-location-l-relative-protospacer" for s in ['pbs', 'rtt', 'rha']] + [f"{s}-melting-temperature" for s in ['edit', 'extension', 'preedit', 'protospacer', 'rha', 'pbs']] + [f"{s}-minimum-free-energy" for s in ['extension', 'extension-scaffold', 'pbs', 'spacer', 'spacer-extension-scaffold', 'spacer-scaffold', 'rtt']] + ['group-id', 'editing-efficiency']
    
    data = data.dropna(subset=['wide_initial_target', 'wide_mutated_target'])
    

    for cell in cell_lines:
        output = []
    
        group_prev = -1
        group_id = -1
        
        # extract the important information
        for ind, item in tqdm.tqdm(data.iterrows(), total=len(data)):
            if not item[cell]: continue
            
            if item['group'] != group_prev:
                group_id += 1
                group_prev = item['group']
            wt_sequence = item['wide_initial_target']
            mut_sequence = item['wide_mutated_target']
            
            # skip if wt-sequence or mut-sequence is empty
            if not wt_sequence or not mut_sequence:
                continue
            
            mut_type = item['Correction_Type']
            if mut_type == 'Deletion':
                mut_type = 2
            elif mut_type == 'Insertion':
                mut_type = 1
            else:
                mut_type = 0
                
            protospacer_location = item['protospacerlocation_only_initial']
            protospacer_location = ast.literal_eval(protospacer_location)
            pbs_location = item['PBSlocation']
            pbs_location = ast.literal_eval(pbs_location)
            rt_wt = item['RT_initial_location']
            rt_wt = ast.literal_eval(rt_wt)     
            
            mts = [item['edited_base_mt'], item['extensionmt'], item['original_base_mt'], item['protospacermt'], item['PBSmt'], item['RToverhangmt']]   
            mfes = [item['MFE_extension'], item['MFE_extension_scaffold'], item['MFE_pbs'], item['MFE_protospacer'], item['MFE_protospacer_extension_scaffold'], item['MFE_protospacer_scaffold'], item['MFE_rt']]
            
            wt_sequence, mut_sequence = align_wt_mut_sequences(wt_sequence, mut_sequence, rt_wt[1], edit_length=item['Correction_Length'], edit_type=mut_type)
            
            output.append([wt_sequence, mut_sequence, item['Correction_Length'], item['PBSlength'], item['RTTlength'], item['RTToverhanglength'], mut_type] + [item['edited_base_mt_nan'], item['original_base_mt_nan'], protospacer_location[0], pbs_location[0] + 1 - protospacer_location[0], rt_wt[0] + 1 - protospacer_location[0], rt_wt[1] - item['RTToverhanglength'] - protospacer_location[0]] + mts + mfes +[group_id, float(item[cell])])
                    
        # split the data into folds
        output_df = pd.DataFrame(output, columns=columns)
        # all numerical columns should be float32
        for col in columns:
            if col not in ['wt-sequence', 'mut-sequence']:
                output_df[col] = output_df[col].astype(np.float32)
        output_df = k_fold_cross_validation_split(output_df, 5)
        output_df.to_csv(pjoin( f"pd-pd-{cell_lines[cell]}-pe2.csv"), index=False)
    

def convert_to_SHAP(source: str) -> None:
    '''
    convert from standard format to SHAP format used for SHAP analysis
    '''
    target = f"shap-{'-'.join(source.split('-')[1:])}"
    if isfile(pjoin('..', 'shap', target)):
        return

    feature_list = ['edit-type-' + s for s in ['replacement', 'insertion', 'deletion']] 
    feature_list += ['gc-content-' + s for s in ['spacer', 'pbs', 'extension', 'rha']]
    feature_list += ['gc-count-' + s for s in ['spacer', 'pbs', 'extension', 'rha']]
    feature_list += ['melting-temperature-' + s for s in ['spacer', 'pbs', 'extension', 'rha']]
    feature_list += ['minimum-free-energy-' + s for s in ['spacer', 'pbs', 'extension', 'rha']]
    feature_list += ['a-at-protospacer-position-' + str(i) for i in range(1, 21)]
    feature_list += ['g-at-protospacer-position-' + str(i) for i in range(1, 21)]
    feature_list += ['t-at-protospacer-position-' + str(i) for i in range(1, 21)]
    feature_list += ['c-at-protospacer-position-' + str(i) for i in range(1, 21)]
    feature_list += [f'{s}-before-edit-position' for s in ['a', 'g', 't', 'c']]
    feature_list += [f'{s}-after-edit-position' for s in ['a', 'g', 't', 'c']]
    feature_list += [f'{s}-length' for s in ['pbs', 'rha', 'lha', 'edit']]
    feature_list += ['pam-disrupted']
    feature_list += [f'maximal-length-of-consecutive-{s}-sequence' for s in ['a', 'g', 't', 'c']]

    feature_list.extend(['spcas9-score', 'group-id', 'editing-efficiency', 'fold'])

    # load the data
    data = pd.read_csv(source)

    output = []

    for ind, item in tqdm.tqdm(data.iterrows(), total=len(data)): 
        wt_sequence = item['wt-sequence']
        mut_sequence = item['mut-sequence']

        pbs = wt_sequence[item['pbs-location-l']:item['pbs-location-r']]
        pbs = get_compliment_dna_to_dna(pbs)
        pbs = get_compliment_dna_to_rna(pbs)
        protospacer = wt_sequence[item['protospacer-location-l']:item['protospacer-location-r']]
        spacer = get_compliment_dna_to_rna(protospacer)
        extension = wt_sequence[item['pbs-location-r']:item['rtt-location-r']]
        extension = get_compliment_dna_to_dna(extension)
        extension = get_compliment_dna_to_rna(extension)
        rha = wt_sequence[item['rha-location-l']:item['rha-location-r']]
        rha = get_compliment_dna_to_dna(rha)
        rha = get_compliment_dna_to_rna(rha)
        lha = wt_sequence[item['lha-location-l']:item['lha-location-r']]
        lha = get_compliment_dna_to_dna(lha)
        lha = get_compliment_dna_to_rna(lha)
        cDNA = mut_sequence[item['pbs-location-l']:item['rtt-location-r']]
        cDNA = get_compliment_dna_to_dna(cDNA)


        # gc content and count
        gc_content_spacer, gc_count_spacer = get_gc_content_and_count(spacer)
        gc_content_pbs, gc_count_pbs = get_gc_content_and_count(pbs)
        gc_content_extension, gc_count_extension = get_gc_content_and_count(extension)
        gc_content_rha, gc_count_rha = get_gc_content_and_count(rha)

        # melting temperature
        melting_temperature_spacer = get_melting_temperature(spacer, table='R_DNA_NN1', c_seq=get_compliment_rna_to_dna(spacer))
        melting_temperature_pbs = get_melting_temperature(pbs, table='R_DNA_NN1', c_seq=get_compliment_rna_to_dna(pbs))
        melting_temperature_extension = get_melting_temperature(extension, table='R_DNA_NN1', c_seq=get_compliment_rna_to_dna(extension))
        melting_temperature_rha = get_melting_temperature(rha, table='R_DNA_NN1', c_seq=get_compliment_rna_to_dna(rha))

        # minimum free energy
        minimum_free_energy_spacer = get_minimum_free_energy(spacer)
        minimum_free_energy_pbs = get_minimum_free_energy(pbs)
        minimum_free_energy_extension = get_minimum_free_energy(extension)
        minimum_free_energy_rha = get_minimum_free_energy(rha)

        # nucleotide at protospacer position
        a_at_protospacer_position = [0] * 20
        g_at_protospacer_position = [0] * 20
        t_at_protospacer_position = [0] * 20
        c_at_protospacer_position = [0] * 20

        for i in range(20):
            if protospacer[i] == 'A':
                a_at_protospacer_position[i] = 1
            elif protospacer[i] == 'G':
                g_at_protospacer_position[i] = 1
            elif protospacer[i] == 'T':
                t_at_protospacer_position[i] = 1
            elif protospacer[i] == 'C':
                c_at_protospacer_position[i] = 1

        # nucleotide before and after edit position
        a_before_edit_position = 0
        g_before_edit_position = 0
        t_before_edit_position = 0
        c_before_edit_position = 0
        a_after_edit_position = 0
        g_after_edit_position = 0
        t_after_edit_position = 0
        c_after_edit_position = 0

        if wt_sequence[item['lha-location-r']] == 'A':
            a_before_edit_position = 1
        elif wt_sequence[item['lha-location-r']] == 'G':
            g_before_edit_position = 1
        elif wt_sequence[item['lha-location-r']] == 'T':
            t_before_edit_position = 1
        elif wt_sequence[item['lha-location-r']] == 'C':
            c_before_edit_position = 1

        if wt_sequence[item['rha-location-l']] == 'A':
            a_after_edit_position = 1
        elif wt_sequence[item['rha-location-l']] == 'G':
            g_after_edit_position = 1
        elif wt_sequence[item['rha-location-l']] == 'T':
            t_after_edit_position = 1
        elif wt_sequence[item['rha-location-l']] == 'C':
            c_after_edit_position = 1

        # length of the sequences
        edit_type = item['mut-type']
        edit_len = item['edit-len']
        rha_len = len(rha)
        lha_len = len(lha)
        pbs_len = len(pbs)

        # pam disrupted
        pam_disrupted = not (wt_sequence[item['protospacer-location-r']:item['protospacer-location-r']+3] == mut_sequence[item['protospacer-location-r']:item['protospacer-location-r']+3])

        # maximal length of consecutive nucleotide sequence
        if len(get_consecutive_n_sequences('A', cDNA) + get_consecutive_n_sequences('A', protospacer)) > 1:
            a_max_length = max([len(seq) for seq in get_consecutive_n_sequences('A', cDNA) + get_consecutive_n_sequences('A', protospacer)])
        else:
            a_max_length = 0
        if len(get_consecutive_n_sequences('G', cDNA) + get_consecutive_n_sequences('G', protospacer)) > 1:
            g_max_length = max([len(seq) for seq in get_consecutive_n_sequences('G', cDNA) + get_consecutive_n_sequences('G', protospacer)])
        else:
            g_max_length = 0
        if len(get_consecutive_n_sequences('T', cDNA) + get_consecutive_n_sequences('T', protospacer)) > 1:
            t_max_length = max([len(seq) for seq in get_consecutive_n_sequences('T', cDNA) + get_consecutive_n_sequences('T', protospacer)])
        else:
            t_max_length = 0
        if len(get_consecutive_n_sequences('C', cDNA) + get_consecutive_n_sequences('C', protospacer)) > 1:
            c_max_length = max([len(seq) for seq in get_consecutive_n_sequences('C', cDNA) + get_consecutive_n_sequences('C', protospacer)])
        else:
            c_max_length = 0
        
        spcas9_score = item['spcas9-score']
        editing_efficiency = item['editing-efficiency']

        output.append([edit_type == 0, edit_type == 1, edit_type == 2, gc_content_spacer, gc_content_pbs, gc_content_extension, gc_content_rha, gc_count_spacer, gc_count_pbs, gc_count_extension, gc_count_rha, melting_temperature_spacer, melting_temperature_pbs, melting_temperature_extension, melting_temperature_rha, minimum_free_energy_spacer, minimum_free_energy_pbs, minimum_free_energy_extension, minimum_free_energy_rha] + a_at_protospacer_position + g_at_protospacer_position + t_at_protospacer_position + c_at_protospacer_position + [a_before_edit_position, g_before_edit_position, t_before_edit_position, c_before_edit_position, a_after_edit_position, g_after_edit_position, t_after_edit_position, c_after_edit_position, pbs_len, rha_len, lha_len, edit_len, pam_disrupted, a_max_length, g_max_length, t_max_length, c_max_length, spcas9_score, item['group-id'], editing_efficiency, item['fold']])

    # save the extracted information
    output_df = pd.DataFrame(output, columns=feature_list, dtype=np.float16)
    output_df.to_csv(pjoin('..', 'shap', target), index=False)
    
def convert_to_shap_1bp(source: str) -> None:
    '''
    convert from standard format to SHAP format used for SHAP analysis
    using only 1bp edit
    '''
    target = f"shap_1bp-{'-'.join(source.split('-')[1:])}"
    if isfile(pjoin('..', 'shap', target)):
        return

    feature_list = ['edit-type-' + s for s in ['replacement', 'insertion', 'deletion']] 
    feature_list += ['gc-content-' + s for s in ['spacer', 'pbs', 'extension', 'rha']]
    feature_list += ['gc-count-' + s for s in ['spacer', 'pbs', 'extension', 'rha']]
    feature_list += ['melting-temperature-' + s for s in ['spacer', 'pbs', 'extension', 'rha']]
    feature_list += ['minimum-free-energy-' + s for s in ['spacer', 'pbs', 'extension', 'rha']]
    feature_list += ['a-at-protospacer-position-' + str(i) for i in range(1, 21)]
    feature_list += ['g-at-protospacer-position-' + str(i) for i in range(1, 21)]
    feature_list += ['t-at-protospacer-position-' + str(i) for i in range(1, 21)]
    feature_list += ['c-at-protospacer-position-' + str(i) for i in range(1, 21)]
    feature_list += [f'{s}-before-edit-position' for s in ['a', 'g', 't', 'c']]
    feature_list += [f'{s}-after-edit-position' for s in ['a', 'g', 't', 'c']]
    feature_list += [f'{s}-length' for s in ['pbs', 'rha', 'lha', 'edit']]
    feature_list += ['pam-disrupted']
    feature_list += [f'maximal-length-of-consecutive-{s}-sequence' for s in ['a', 'g', 't', 'c']]
    feature_list += [f'pre-edit-base-{s}' for s in ['a', 'g', 't', 'c']]
    feature_list += [f'post-edit-base-{s}' for s in ['a', 'g', 't', 'c']]

    feature_list.extend(['spcas9-score', 'group-id', 'editing-efficiency', 'fold'])

    # load the data
    data = pd.read_csv(source)
    # only 1bp edit
    data = data[data['edit-len'] == 1]

    output = []

    for ind, item in tqdm.tqdm(data.iterrows(), total=len(data)): 
        wt_sequence = item['wt-sequence']
        mut_sequence = item['mut-sequence']

        pbs = wt_sequence[item['pbs-location-l']:item['pbs-location-r']]
        pbs = get_compliment_dna_to_dna(pbs)
        pbs = get_compliment_dna_to_rna(pbs)
        protospacer = wt_sequence[item['protospacer-location-l']:item['protospacer-location-r']]
        spacer = get_compliment_dna_to_rna(protospacer)
        extension = wt_sequence[item['pbs-location-r']:item['rtt-location-r']]
        extension = get_compliment_dna_to_dna(extension)
        extension = get_compliment_dna_to_rna(extension)
        rha = wt_sequence[item['rha-location-l']:item['rha-location-r']]
        rha = get_compliment_dna_to_dna(rha)
        rha = get_compliment_dna_to_rna(rha)
        lha = wt_sequence[item['lha-location-l']:item['lha-location-r']]
        lha = get_compliment_dna_to_dna(lha)
        lha = get_compliment_dna_to_rna(lha)
        cDNA = mut_sequence[item['pbs-location-l']:item['rtt-location-r']]
        cDNA = get_compliment_dna_to_dna(cDNA)


        # gc content and count
        gc_content_spacer, gc_count_spacer = get_gc_content_and_count(spacer)
        gc_content_pbs, gc_count_pbs = get_gc_content_and_count(pbs)
        gc_content_extension, gc_count_extension = get_gc_content_and_count(extension)
        gc_content_rha, gc_count_rha = get_gc_content_and_count(rha)

        # melting temperature
        melting_temperature_spacer = get_melting_temperature(spacer, table='R_DNA_NN1', c_seq=get_compliment_rna_to_dna(spacer))
        melting_temperature_pbs = get_melting_temperature(pbs, table='R_DNA_NN1', c_seq=get_compliment_rna_to_dna(pbs))
        melting_temperature_extension = get_melting_temperature(extension, table='R_DNA_NN1', c_seq=get_compliment_rna_to_dna(extension))
        melting_temperature_rha = get_melting_temperature(rha, table='R_DNA_NN1', c_seq=get_compliment_rna_to_dna(rha))

        # minimum free energy
        minimum_free_energy_spacer = get_minimum_free_energy(spacer)
        minimum_free_energy_pbs = get_minimum_free_energy(pbs)
        minimum_free_energy_extension = get_minimum_free_energy(extension)
        minimum_free_energy_rha = get_minimum_free_energy(rha)

        # nucleotide at protospacer position
        a_at_protospacer_position = [0] * 20
        g_at_protospacer_position = [0] * 20
        t_at_protospacer_position = [0] * 20
        c_at_protospacer_position = [0] * 20

        for i in range(20):
            if protospacer[i] == 'A':
                a_at_protospacer_position[i] = 1
            elif protospacer[i] == 'G':
                g_at_protospacer_position[i] = 1
            elif protospacer[i] == 'T':
                t_at_protospacer_position[i] = 1
            elif protospacer[i] == 'C':
                c_at_protospacer_position[i] = 1

        # nucleotide before and after edit position
        a_before_edit_position = 0
        g_before_edit_position = 0
        t_before_edit_position = 0
        c_before_edit_position = 0
        a_after_edit_position = 0
        g_after_edit_position = 0
        t_after_edit_position = 0
        c_after_edit_position = 0

        if wt_sequence[item['lha-location-r']] == 'A':
            a_before_edit_position = 1
        elif wt_sequence[item['lha-location-r']] == 'G':
            g_before_edit_position = 1
        elif wt_sequence[item['lha-location-r']] == 'T':
            t_before_edit_position = 1
        elif wt_sequence[item['lha-location-r']] == 'C':
            c_before_edit_position = 1

        if wt_sequence[item['rha-location-l']] == 'A':
            a_after_edit_position = 1
        elif wt_sequence[item['rha-location-l']] == 'G':
            g_after_edit_position = 1
        elif wt_sequence[item['rha-location-l']] == 'T':
            t_after_edit_position = 1
        elif wt_sequence[item['rha-location-l']] == 'C':
            c_after_edit_position = 1

        # length of the sequences
        edit_type = item['mut-type']
        edit_len = item['edit-len']
        rha_len = len(rha)
        lha_len = len(lha)
        pbs_len = len(pbs)
        
        pre_edit_base = wt_sequence[item['lha-location-r']]
        post_edit_base = mut_sequence[item['lha-location-r']]
        pre_edit_bases = [0 for _ in range(4)]
        post_edit_bases = [0 for _ in range(4)]
        
        if pre_edit_base == 'A':
            pre_edit_bases[0] = 1
        elif pre_edit_base == 'G':
            pre_edit_bases[1] = 1
        elif pre_edit_base == 'T':
            pre_edit_bases[2] = 1
        elif pre_edit_base == 'C':
            pre_edit_bases[3] = 1
        if post_edit_base == 'A':
            post_edit_bases[0] = 1
        elif post_edit_base == 'G':
            post_edit_bases[1] = 1
        elif post_edit_base == 'T':
            post_edit_bases[2] = 1
        elif post_edit_base == 'C':
            post_edit_bases[3] = 1
        

        # pam disrupted
        pam_disrupted = not (wt_sequence[item['protospacer-location-r']:item['protospacer-location-r']+3] == mut_sequence[item['protospacer-location-r']:item['protospacer-location-r']+3])

        # maximal length of consecutive nucleotide sequence
        if len(get_consecutive_n_sequences('A', cDNA) + get_consecutive_n_sequences('A', protospacer)) > 1:
            a_max_length = max([len(seq) for seq in get_consecutive_n_sequences('A', cDNA) + get_consecutive_n_sequences('A', protospacer)])
        else:
            a_max_length = 0
        if len(get_consecutive_n_sequences('G', cDNA) + get_consecutive_n_sequences('G', protospacer)) > 1:
            g_max_length = max([len(seq) for seq in get_consecutive_n_sequences('G', cDNA) + get_consecutive_n_sequences('G', protospacer)])
        else:
            g_max_length = 0
        if len(get_consecutive_n_sequences('T', cDNA) + get_consecutive_n_sequences('T', protospacer)) > 1:
            t_max_length = max([len(seq) for seq in get_consecutive_n_sequences('T', cDNA) + get_consecutive_n_sequences('T', protospacer)])
        else:
            t_max_length = 0
        if len(get_consecutive_n_sequences('C', cDNA) + get_consecutive_n_sequences('C', protospacer)) > 1:
            c_max_length = max([len(seq) for seq in get_consecutive_n_sequences('C', cDNA) + get_consecutive_n_sequences('C', protospacer)])
        else:
            c_max_length = 0
        
        spcas9_score = item['spcas9-score']
        editing_efficiency = item['editing-efficiency']

        output.append([edit_type == 0, edit_type == 1, edit_type == 2, gc_content_spacer, gc_content_pbs, gc_content_extension, gc_content_rha, gc_count_spacer, gc_count_pbs, gc_count_extension, gc_count_rha, melting_temperature_spacer, melting_temperature_pbs, melting_temperature_extension, melting_temperature_rha, minimum_free_energy_spacer, minimum_free_energy_pbs, minimum_free_energy_extension, minimum_free_energy_rha] + a_at_protospacer_position + g_at_protospacer_position + t_at_protospacer_position + c_at_protospacer_position + [a_before_edit_position, g_before_edit_position, t_before_edit_position, c_before_edit_position, a_after_edit_position, g_after_edit_position, t_after_edit_position, c_after_edit_position, pbs_len, rha_len, lha_len, edit_len, pam_disrupted, a_max_length, g_max_length, t_max_length, c_max_length] + pre_edit_bases + post_edit_bases + [spcas9_score, item['group-id'], editing_efficiency, item['fold']])

    # save the extracted information according to edit type
    output_df = pd.DataFrame(output, columns=feature_list, dtype=np.float16)
    output_df.to_csv(pjoin('shap', target), index=False)
    output_df_replace = output_df[output_df['edit-type-replacement'] == 1]
    output_df_insert = output_df[output_df['edit-type-insertion'] == 1]
    output_df_delete = output_df[output_df['edit-type-deletion'] == 1]
    output_df_replace.to_csv(pjoin('shap', target.split('.')[0] + '-replace.csv'), index=False)
    output_df_insert.to_csv(pjoin('shap', target.split('.')[0] + '-insert.csv'), index=False)
    output_df_delete.to_csv(pjoin('shap', target.split('.')[0] + '-delete.csv'), index=False)
    

def convert_to_conventional_ml(source: str) -> pd.DataFrame:
    '''
    convert from std format to conventional machine learning format
    '''
    target = f"ml-{'-'.join(source.split('-')[1:])}"
    if isfile(pjoin('conventional-ml', target)):
        return
    
    features = ['gc-count-pbs', 'rha-length', 'pbs-length', 'gc-count-extension',
       'melting-temperature-pbs', 'spcas9-score', 'lha-length',
       'maximal-length-of-consecutive-t-sequence', 'edit-type-replacement',
       'g-at-protospacer-position-16', 'gc-content-rha', 'pam-disrupted',
       'edit-length', 'c-at-protospacer-position-17',
       'melting-temperature-rha', 'minimum-free-energy-extension',
       'a-at-protospacer-position-13', 'a-at-protospacer-position-14',
       't-at-protospacer-position-16', 'gc-content-extension', 'gc-count-rha',
       'g-at-protospacer-position-15', 'g-at-protospacer-position-19',
       'gc-content-pbs']
    
    features += ['group-id', 'editing-efficiency']
    
    pe = target.split('-')[-1].split('.')[0]
    
    # R is A or G, H is A, C or T, N is A, C, G or T
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
    
    pam = pam_table[pe]
    
    # load the data
    data = pd.read_csv(source)
    
    output = []
    
    for ind, item in tqdm.tqdm(data.iterrows(), total=len(data)):
        wt_sequence = item['wt-sequence']
        mut_sequence = item['mut-sequence']

        pbs = wt_sequence[item['pbs-location-l']:item['pbs-location-r']]
        pbs = get_compliment_dna_to_dna(pbs)
        pbs = get_compliment_dna_to_rna(pbs)
        protospacer = wt_sequence[item['protospacer-location-l']:item['protospacer-location-r']]
        spacer = get_compliment_dna_to_rna(protospacer)
        extension = wt_sequence[item['pbs-location-r']:item['rtt-location-r']]
        extension = get_compliment_dna_to_dna(extension)
        extension = get_compliment_dna_to_rna(extension)
        rha = wt_sequence[item['rha-location-l']:item['rha-location-r']]
        rha = get_compliment_dna_to_dna(rha)
        rha = get_compliment_dna_to_rna(rha)
        lha = wt_sequence[item['lha-location-l']:item['lha-location-r']]
        lha = get_compliment_dna_to_dna(lha)
        lha = get_compliment_dna_to_rna(lha)
        cDNA = mut_sequence[item['pbs-location-l']:item['rtt-location-r']]
        cDNA = get_compliment_dna_to_dna(cDNA)
        
        # gc content and count
        gc_count_pbs, _ = get_gc_content_and_count(pbs)
        gc_count_spacer, _ = get_gc_content_and_count(spacer)
        gc_count_extension, _ = get_gc_content_and_count(extension)
        gc_count_rha, _ = get_gc_content_and_count(rha)
        
        # gc content and count
        gc_content_spacer, gc_count_spacer = get_gc_content_and_count(spacer)
        gc_content_pbs, gc_count_pbs = get_gc_content_and_count(pbs)
        gc_content_extension, gc_count_extension = get_gc_content_and_count(extension)
        gc_content_rha, gc_count_rha = get_gc_content_and_count(rha)

        # melting temperature
        melting_temperature_spacer = get_melting_temperature(spacer, table='R_DNA_NN1')
        melting_temperature_pbs = get_melting_temperature(pbs, table='R_DNA_NN1')
        melting_temperature_extension = get_melting_temperature(extension, table='R_DNA_NN1')
        melting_temperature_rha = get_melting_temperature(rha, table='R_DNA_NN1')

        # minimum free energy
        minimum_free_energy_spacer = get_minimum_free_energy(spacer)
        minimum_free_energy_pbs = get_minimum_free_energy(pbs)
        minimum_free_energy_extension = get_minimum_free_energy(extension)
        minimum_free_energy_rha = get_minimum_free_energy(rha)

        # nucleotide at protospacer position
        a_at_protospacer_position = [0] * 20
        g_at_protospacer_position = [0] * 20
        t_at_protospacer_position = [0] * 20
        c_at_protospacer_position = [0] * 20

        for i in range(20):
            if protospacer[i] == 'A':
                a_at_protospacer_position[i] = 1
            elif protospacer[i] == 'G':
                g_at_protospacer_position[i] = 1
            elif protospacer[i] == 'T':
                t_at_protospacer_position[i] = 1
            elif protospacer[i] == 'C':
                c_at_protospacer_position[i] = 1


        # length of the sequences
        edit_type = item['mut-type']
        edit_len = item['edit-len']
        rha_len = len(rha)
        lha_len = len(lha)
        pbs_len = len(pbs)

        # pam disrupted
        pam_disrupted = not match_pam(mut_sequence[item['protospacer-location-r']:item['protospacer-location-r']+len(pam)], pam)

        # maximal length of consecutive nucleotide sequence
        if len(get_consecutive_n_sequences('T', cDNA) + get_consecutive_n_sequences('T', protospacer)) > 1:
            t_max_length = max([len(seq) for seq in get_consecutive_n_sequences('T', cDNA) + get_consecutive_n_sequences('T', protospacer)])
        else:
            t_max_length = 0
        
        spcas9_score = item['spcas9-score']
        editing_efficiency = item['editing-efficiency']

        output.append([gc_count_pbs, rha_len, pbs_len, gc_count_extension, melting_temperature_pbs, spcas9_score, lha_len, t_max_length, edit_type == 0, g_at_protospacer_position[15], gc_content_rha, pam_disrupted, edit_len, c_at_protospacer_position[16], melting_temperature_rha, minimum_free_energy_extension, a_at_protospacer_position[12], a_at_protospacer_position[13], t_at_protospacer_position[15], gc_content_extension, gc_count_rha, g_at_protospacer_position[14], g_at_protospacer_position[18], gc_content_pbs, item['group-id'], editing_efficiency])
    
    # save the extracted information
    output_df = pd.DataFrame(output, columns=features, dtype=np.float32)
    # add fold column
    output_df = k_fold_cross_validation_split(output_df, 5)
    # save the data
    output_df.to_csv(pjoin('..', 'conventional-ml', target), index=False)
    
    return output_df

def convert_to_ensemble_df(data: pd.DataFrame) -> pd.DataFrame:
    """ Convert loaded std data to ensemble data format

    """
    features = ['wt_sequence', 'mut_sequence', 'gc-count-pbs', 'rha-length', 'pbs-length', 'gc-count-extension',
       'melting-temperature-pbs', 'spcas9-score', 'lha-length',
       'maximal-length-of-consecutive-t-sequence', 'edit-type-replacement',
       'g-at-protospacer-position-16', 'gc-content-rha', 'pam-disrupted',
       'edit-length', 'c-at-protospacer-position-17',
       'melting-temperature-rha', 'minimum-free-energy-extension',
       'a-at-protospacer-position-13', 'a-at-protospacer-position-14',
       't-at-protospacer-position-16', 'gc-content-extension', 'gc-count-rha',
       'g-at-protospacer-position-15', 'g-at-protospacer-position-19',
       'gc-content-pbs']
    
    features += ['group-id', 'editing-efficiency']
    
    output = []
    
    for ind, item in tqdm.tqdm(data.iterrows(), total=len(data)):
        wt_sequence = item['wt-sequence']
        mut_sequence = item['mut-sequence']

        pbs = wt_sequence[item['pbs-location-l']:item['pbs-location-r']]
        pbs = get_compliment_dna_to_dna(pbs)
        pbs = get_compliment_dna_to_rna(pbs)
        protospacer = wt_sequence[item['protospacer-location-l']:item['protospacer-location-r']]
        spacer = get_compliment_dna_to_rna(protospacer)
        extension = wt_sequence[item['pbs-location-r']:item['rtt-location-r']]
        extension = get_compliment_dna_to_dna(extension)
        extension = get_compliment_dna_to_rna(extension)
        rha = wt_sequence[item['rha-location-l']:item['rha-location-r']]
        rha = get_compliment_dna_to_dna(rha)
        rha = get_compliment_dna_to_rna(rha)
        lha = wt_sequence[item['lha-location-l']:item['lha-location-r']]
        lha = get_compliment_dna_to_dna(lha)
        lha = get_compliment_dna_to_rna(lha)
        cDNA = mut_sequence[item['pbs-location-l']:item['rtt-location-r']]
        cDNA = get_compliment_dna_to_dna(cDNA)

        # mask out regions outside of pbs and rtt for mutated sequence
        mut_sequence = 'N' * item['pbs-location-l'] + mut_sequence[item['pbs-location-l']:item['rtt-location-r']] + 'N' * (len(mut_sequence) - item['rtt-location-r'])
        
        # gc content and count
        gc_count_pbs, _ = get_gc_content_and_count(pbs)
        gc_count_spacer, _ = get_gc_content_and_count(spacer)
        gc_count_extension, _ = get_gc_content_and_count(extension)
        gc_count_rha, _ = get_gc_content_and_count(rha)
        
        # gc content and count
        gc_content_spacer, gc_count_spacer = get_gc_content_and_count(spacer)
        gc_content_pbs, gc_count_pbs = get_gc_content_and_count(pbs)
        gc_content_extension, gc_count_extension = get_gc_content_and_count(extension)
        gc_content_rha, gc_count_rha = get_gc_content_and_count(rha)

        # melting temperature
        melting_temperature_spacer = get_melting_temperature(spacer, table='R_DNA_NN1')
        melting_temperature_pbs = get_melting_temperature(pbs, table='R_DNA_NN1')
        melting_temperature_extension = get_melting_temperature(extension, table='R_DNA_NN1')
        melting_temperature_rha = get_melting_temperature(rha, table='R_DNA_NN1')

        # minimum free energy
        minimum_free_energy_spacer = get_minimum_free_energy(spacer)
        minimum_free_energy_pbs = get_minimum_free_energy(pbs)
        minimum_free_energy_extension = get_minimum_free_energy(extension)
        minimum_free_energy_rha = get_minimum_free_energy(rha)

        # nucleotide at protospacer position
        a_at_protospacer_position = [0] * 20
        g_at_protospacer_position = [0] * 20
        t_at_protospacer_position = [0] * 20
        c_at_protospacer_position = [0] * 20

        for i in range(20):
            if protospacer[i] == 'A':
                a_at_protospacer_position[i] = 1
            elif protospacer[i] == 'G':
                g_at_protospacer_position[i] = 1
            elif protospacer[i] == 'T':
                t_at_protospacer_position[i] = 1
            elif protospacer[i] == 'C':
                c_at_protospacer_position[i] = 1


        # length of the sequences
        edit_type = item['mut-type']
        edit_len = item['edit-len']
        rha_len = len(rha)
        lha_len = len(lha)
        pbs_len = len(pbs)

        # pam disrupted
        pam_disrupted = match_pam

        # maximal length of consecutive nucleotide sequence
        if len(get_consecutive_n_sequences('T', cDNA) + get_consecutive_n_sequences('T', protospacer)) > 1:
            t_max_length = max([len(seq) for seq in get_consecutive_n_sequences('T', cDNA) + get_consecutive_n_sequences('T', protospacer)])
        else:
            t_max_length = 0
        
        # spcas9_score = item['spcas9-score']
        # spcas9_score needs to be calculated

        editing_efficiency = item['editing-efficiency']

        output.append([wt_sequence, mut_sequence, gc_count_pbs, rha_len, pbs_len, gc_count_extension, melting_temperature_pbs, spcas9_score, lha_len, t_max_length, edit_type == 0, g_at_protospacer_position[15], gc_content_rha, pam_disrupted, edit_len, c_at_protospacer_position[16], melting_temperature_rha, minimum_free_energy_extension, a_at_protospacer_position[12], a_at_protospacer_position[13], t_at_protospacer_position[15], gc_content_extension, gc_count_rha, g_at_protospacer_position[14], g_at_protospacer_position[18], gc_content_pbs, item['group-id'], editing_efficiency])

    # save the extracted information
    output_df = pd.DataFrame(output, columns=features, dtype=np.float32)
    # join output_df with data
    output_df = pd.concat([data, output_df], axis=1)
    # move group_id and editing_efficiency to the end
    output_df = output_df[[col for col in output_df.columns if col not in ['group-id', 'editing-efficiency']] + ['group-id', 'editing-efficiency']]
    return output_df

# =============================================================================
# Sequence data preprocessing
# =============================================================================

def onehot_encode(seq: str) -> np.ndarray:
    '''
    One hot encode a DNA sequence into a 4 x n numpy array
    '''
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


def onehot_encode_series(series: pd.Series) -> torch.Tensor:
    '''
    One hot encode a series of DNA sequences
    '''
    n = len(series)
    m = len(series.iloc[0])
    encoding = np.zeros((n, 4, m), dtype=np.float32)
    for i, seq in enumerate(series):
        encoding[i] = onehot_encode(seq)
    return encoding

class GeneFeatureData:
    '''
    Dataset class for the gene + feature data
    '''
    def __init__(self, g, x, target):
        self.g = g
        self.x = x
        self.target = target

    def __len__(self):
        return len(self.target)
    
    def __getitem__(self, idx):
        return self.g[idx], self.x[idx], self.target[idx]
    

def get_compliment_dna_to_rna(sequence: str) -> str:
    '''
    Returns the RNA compliment of a given DNA sequence
    '''
    # remove Ns
    if 'N' in sequence:
        sequence = sequence.replace('N', '')
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
            print(f'Invalid base: {base}')
    return rna_compliment

def get_compliment_rna_to_dna(sequence: str) -> str:
    '''
    Returns the DNA compliment of a given RNA sequence
    '''
    # remove Ns
    if 'N' in sequence:
        sequence = sequence.replace('N', '')
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
            print(f'Invalid base: {base}')
    return dna_compliment

def get_compliment_dna_to_dna(sequence: str) -> str:
    '''
    Returns the DNA compliment of a given DNA sequence
    '''
    # remove Ns
    if 'N' in sequence:
        sequence = sequence.replace('N', '')
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
            print(f'Invalid base: {base}')
    return dna_compliment

def align_wt_mut_sequences(wt_sequence: str, mut_sequence: str, edit_position: int, edit_length: int, edit_type: int) -> Tuple[str, str]:
    '''
    Align the wild type and mutated sequences
    '''
    l = len(wt_sequence)
    if edit_type == 1: # insertion
        wt_sequence = wt_sequence[:edit_position] + 'N'*edit_length + wt_sequence[edit_position:]
    elif edit_type == 2: # deletion
        mut_sequence = mut_sequence[:edit_position] + 'N'*edit_length + mut_sequence[edit_position:]
        
    # make sure the sequences are of the same length
    wt_sequence = wt_sequence[:l]
    mut_sequence = mut_sequence[:l]
    
    return wt_sequence, mut_sequence
    
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
    match = True
    for i in range(len(pam)):
        if pam[i] == 'N':
            continue
        elif pam[i] == 'R':
            if sequence[i] not in ['A', 'G']:
                match = False
                break
        elif pam[i] == 'H':
            if sequence[i] not in ['A', 'C', 'T']:
                match = False
                break
        else:
            if sequence[i] != pam[i]:
                match = False
                break
    return match

# =============================================================================
# Training Data Preprocessing
# =============================================================================

def k_fold_cross_validation_split(data: pd.DataFrame, fold: int) -> pd.DataFrame:
    '''
    K fold cross validation, ensuring data with the same group-id is in the same fold
    '''
    # shuffle the data
    data['fold'] = 0
    for f in range(fold):
        fold_data = data[data['group-id'] % fold == f]
        data.loc[fold_data.index, 'fold'] = f
    
    return data