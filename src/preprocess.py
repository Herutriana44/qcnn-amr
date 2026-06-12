import tools
from tools import clean_seq, getDAC, list_to_string, string_to_list
import pandas as pd
import re
import os
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(current_dir)

ALGORITHM_NAME = "QCNN_Hybrid_Weighted"
output_folder = f"{ALGORITHM_NAME}-result"

data_dir = os.path.join(PROJECT_ROOT, 'data')
result_dir = os.path.join(PROJECT_ROOT, output_folder)

os.makedirs(result_dir, exist_ok=True)

def run():
    df = pd.read_csv(os.path.join(data_dir, 'aro_categories_index.tsv'), delimiter='\t')
    df_seq = tools.seq_to_df(os.path.join(data_dir, "nucleotide_fasta_protein_homolog_model.fasta"))
    df_seq.to_excel(os.path.join(result_dir, "nucleotide_fasta_protein_homolog_model.xlsx"),index=False)
    # explode id to many data and filter only 'Escherichia coli'
    split_id_data = df_seq['id'].str.split('|', expand=True)
    split_id_data.columns = ['id_source', 'accession_id', 'strand', 'coordinates', 'aro_id', 'gene_name']
    df_seq = df_seq.drop(columns=['id'])
    df_seq = pd.concat([df_seq, split_id_data], axis=1)
    df_seq['species'] = df_seq['description'].apply(lambda x: re.search(r'\[(.*?)\]', x).group(1) if re.search(r'\[(.*?)\]', x) else None)
    df_seq = df_seq.dropna(subset=['Resistance Mechanism'])
    df_seq = df_seq.reset_index(drop=True)

    df_seq['species'].value_counts().head(5).plot(kind='bar')
    plt.savefig(os.path.join(result_dir, "top_5_species_bar_graph.png"), dpi=300, bbox_inches="tight")
    
    df_seq = df_seq[df_seq['species'] == 'Escherichia coli']
    # Create a dictionary mapping accessions (both protein and DNA) to resistance mechanisms
    accession_to_resistance = {}

    # Map Protein Accession to Resistance Mechanism from df
    protein_map = df.set_index('Protein Accession')['Resistance Mechanism'].to_dict()
    accession_to_resistance.update(protein_map)

    # Map DNA Accession to Resistance Mechanism from df
    # Only add if the accession is not already in the dictionary to avoid overwriting
    dna_map = df.set_index('DNA Accession')['Resistance Mechanism'].to_dict()
    for dna_acc, resistance_mech in dna_map.items():
        if dna_acc not in accession_to_resistance:
            accession_to_resistance[dna_acc] = resistance_mech

    # Apply the mapping to create the 'Resistance Mechanism' column in df_seq
    # Using .apply() with .get() to handle potential issues with .map() and ensure it's treating the dict correctly
    df_seq['Resistance Mechanism'] = df_seq['accession_id'].apply(lambda x: accession_to_resistance.get(x, None))
    df_seq['sequence'] = df_seq['sequence'].apply(clean_seq)
    df_seq['dac'] = df_seq['sequence'].apply(getDAC)
    df_seq['dac'] = df_seq['dac'].apply(list_to_string)
    df_seq = df_seq.dropna(subset=['Resistance Mechanism'])
    df_seq = df_seq.reset_index(drop=True)

    df_seq['Resistance Mechanism'].value_counts().plot(kind='bar')
    plt.savefig(os.path.join(result_dir, "resistance mechanism on escherichia coli.png"), dpi=300, bbox_inches="tight")

    x = df_seq['dac'].tolist()
    x = [string_to_list(i) for i in x]
    y = df_seq['Resistance Mechanism'].tolist()
    y_map = {i:idx for idx, i in enumerate(set(y))}
    y = [y_map[i] for i in y]

    num_label = df_seq['Resistance Mechanism'].value_counts().tolist()
    total_label = len(num_label)
    class_weights = [sum(num_label)/(total_label*num_label[i]) for i in range(len(num_label))]
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    return df, df_seq, x, y