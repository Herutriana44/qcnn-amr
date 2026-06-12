from PyBioMed.PyDNA.PyDNAac import GetDAC
from Bio import SeqIO
import re
import pandas as pd

def clean_seq(seq):
  seq = seq.replace('U', 'T')
  # Remove any character that is not A, C, G, or T
  seq = re.sub(r'[^ACGT]', '', seq)
  return seq

def getDAC(seq):
  dac = GetDAC(seq, phyche_index=['Twist','Tilt', 'Roll','Shift', 'Slide', 'Rise'])
  dac = list(dac.values())
  return dac

def list_to_string(lst):
    return ';'.join(map(str, lst))

def string_to_list(s):
    return list(map(float, s.split(';')))

def seq_to_df(fasta_file):
    sequences = []

    for record in SeqIO.parse(fasta_file, 'fasta'):
        sequences.append(record)

    df_seq = pd.DataFrame(columns=['id', 'description', 'sequence'])
    for record in sequences:
        df_seq = pd.concat([df_seq, pd.DataFrame({'id': [record.id], 'description': [record.description], 'sequence': [str(record.seq)]})], ignore_index=True)

    return df_seq