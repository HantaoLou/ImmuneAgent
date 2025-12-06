import os
import pandas as pd
import json
from Bio import PDB
import glob
# import alphafold3

import sys
import argparse
# ===========================================================================================

parser = argparse.ArgumentParser(description='train')
parser.add_argument('--input_file_name', type=str, default='0315_need_3seeds.csv')
parser.add_argument('--file_name_col', type=str, default='clone_id')
parser.add_argument('--antigen_name', type=str, default='TE24_H5N1')
parser.add_argument('--antig_letter', type=str, default='ABC')
parser.add_argument('--heavy_letter', type=str, default='HIJ')
parser.add_argument('--light_letter', type=str, default='LMN')
args = parser.parse_args()

# input_name = 'results_AF3_0110'
# output_name = input_name
# suffix = '.txt'
CONVERT_JSON = True
# CONVERT_JSON = False
ANTIG_LETTER = args.antig_letter
HEAVY_LETTER = args.heavy_letter
LIGHT_LETTER = args.light_letter


gpu_id=0

[input_name,suffix] = args.input_file_name.split('.')
suffix = '.'+suffix
print(f"input_name: {input_name}, suffix: {suffix}")
file_name_col = args.file_name_col

# input_name = 'select_Ab_v2'
# suffix = '.xlsx'
# input_name = '20250120_select_results-filtered'
# suffix = '.csv'
# file_name_col = 'mAb'

# input_name = '20250310_second-batch_AF3'
# input_name = '20250313_second-batch_AF3'
# input_name = '0315_need_1seeds'
# input_name = '0315_need_3seeds'

# suffix = '.xlsx'
# suffix = '.csv'

# file_name_col = 'clone_id'

# input_name = 'protein_mole'
# suffix = '.json'

if not CONVERT_JSON:
    output_name = input_name
    antigen_name = ''
else:
    antigen_name = args.antigen_name
    # antigen_name = 'TE24_H5N1'
    # antigen_name = 'QH05_H5N1'
    # antigen_name = 'IN05_H5N1'
    # antigen_name = 'HB10_H5N1'
    # antigen_name = 'AS20_H5N1'

    output_name = input_name+'_'+antigen_name
    header = [file_name_col, 'Heavy', 'Light', antigen_name]
    HEADER = {
        file_name_col: file_name_col,
        'Heavy': HEAVY_LETTER,
        'Light': LIGHT_LETTER,
        antigen_name: ANTIG_LETTER,
    }
    header = list(HEADER.values())
    print(f"header: {header}")
# ===========================================================================================

ANTIGEN_NAME = antigen_name
ROOT_DIR = '/data/lht/AF3'

PDB_DIR = os.path.join(ROOT_DIR, 'af3_inputs', 'pdb_files')
CSV_DIR = os.path.join(ROOT_DIR, 'af3_inputs', 'csv_files')
JSON_DIR = os.path.join(ROOT_DIR, 'json_files')

MODEL_DIR = os.path.join(ROOT_DIR, 'af3_model')
OUT_DIR = os.path.join(ROOT_DIR, 'af3_outputs')
PUBLIC_DATA_DIR = os.path.join(ROOT_DIR, 'public_databases')


DICT_TEMPLATE = {
"name": "TEMPLATE",
#   "modelSeeds": [3, 34, 37],
"modelSeeds": [5],
"dialect": "alphafold3",
"version": 2
}


def read_table(table_pth, usecols):
    try:
        table = pd.read_csv(table_pth,usecols=usecols)
    except:
        table = pd.read_excel(table_pth,usecols=usecols)
    return table

def read_seq_dict_list_from_csv(csv_file, header=HEADER):
    df = read_table(csv_file, usecols=header.keys())
    df.rename(columns=header, inplace=True)
    # print(df)
    seq_dict_list = df.to_dict(orient='records')
    return seq_dict_list



def read_seq_dict_from_pdb(pdb_file, suffix='.pdb', seq_type='protein'):
    
    print(f"Reading pdb file: {pdb_file}")
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('pdb', pdb_file)
    try:
        model = structure[0]
    except KeyError:
        print(f"No model with ID 0 found in {pdb_file}")
        return None

    seq_dict = DICT_TEMPLATE.copy()
    seq_list = []
    for chain in model:
        chain_id = chain.id
        chain_dict = {}
        chain_dict['id']=chain_id
        # chain_dict['seq_type']=seq_type
        chain_dict['sequence']=''
        for residue in chain:
            if PDB.is_aa(residue, standard=True):
                chain_dict['sequence'] += PDB.Polypeptide.three_to_one(residue.resname)
                # chain_dict[chain_id] += PDB.Polypeptide.three_to_one(residue.resname)
            elif residue.resname in ['A', 'C', 'G', 'T', 'U']:
                chain_dict['sequence'] += residue.resname
            else:
                chain_dict['sequence'] += 'X'
        
        seq_list.append({seq_type:chain_dict})
    seq_dict['sequences'] = seq_list
    return seq_dict

def read_seq_dict_list_from_pdb(pdb_dir, suffix='.pdb'):
    seqs = []
    for file in os.listdir(pdb_dir):
        if file.endswith(suffix):
            pdb_file = os.path.join(pdb_dir, file)
            chain_dict = read_seq_dict_from_pdb(pdb_file, suffix)
            seqs.append(chain_dict)
    return seqs

def convert_seq_dict(
        input_dict, 
        name_col='id', 
        seq_type='protein', 
        # antig_name = ANTIGEN_NAME, 
        # antig_copy_num=ANTIG_COPY_NUM,
        # antib_copy_num=ANTIB_COPY_NUM
        ):
    seq_dict = DICT_TEMPLATE.copy()
    seq_dict['name'] = input_dict[name_col]
    seq_list = []
    # last_key = list(input_dict.keys())[-1]
    # input_dict[last_key] = 'Antigen'
    for key, value in input_dict.items():
        # rec_num = 1
        # if key == antig_name:
        #     key = 'ABC'
        #     rec_num = antig_copy_num
        if key != name_col:
            # for i in range(rec_num):
            for i in range(len(key)):
                seq_list.append(
                    {seq_type:
                        {
                            'id': key[0+i].upper(),
                            'sequence': value
                        }
                    }
                )
    seq_dict['sequences'] = seq_list
    return seq_dict


def convert2afformat(suffix='csv', input_name=input_name, output_name=output_name, header=[]):
    if suffix in ['.pdb', '.cif','txt']:
        input_dir = os.path.join(PDB_DIR, input_name, '*'+suffix)
        output_dir = os.path.join(JSON_DIR, output_name)

        try:
            os.makedirs(output_dir)
            print(f"Created directory: '{output_dir}'")
        except:
            print(f"Directory already exists: '{output_dir}'")

        input_pdb_path = glob.glob(input_dir)
        for pdb_file_path in input_pdb_path:
            # print(pdb_file_path)
            seq_dict = read_seq_dict_from_pdb(pdb_file_path)
            pdb_file_name = os.path.basename(pdb_file_path)
            seq_dict['name'] = pdb_file_name

            json_file = os.path.join(output_dir, pdb_file_name+'.json')
            with open(json_file, 'w') as f:
                json.dump(seq_dict, f, indent=2)
    elif suffix in ['.csv','.xlsx']:
        input_csv_path = [os.path.join(CSV_DIR, input_name+suffix)]
        print(f"input_csv_path: {input_csv_path}")
        # input_csv_path = glob.glob(input_dir)
        output_dir = os.path.join(JSON_DIR, output_name)

        try:
            os.makedirs(output_dir)
            print(f"Created directory: '{output_dir}'")
        except:
            print(f"Directory already exists: '{output_dir}'")
        
        for csv_file_path in input_csv_path:
            seq_dict_list = read_seq_dict_list_from_csv(csv_file_path,HEADER)

            csv_file_name = os.path.basename(csv_file_path)
            for seq_dict in seq_dict_list:
                print(seq_dict)
                json_file = os.path.join(output_dir, seq_dict[header[0]].replace('(',"").replace(')',"")+'.json')
                with open(json_file, 'w') as f:
                    json.dump(convert_seq_dict(seq_dict,header[0]), f, indent=2)


if __name__ == '__main__':
    if CONVERT_JSON:
        convert2afformat(suffix=suffix, input_name=input_name, output_name=output_name, header=header)
    output_dir = os.path.join(OUT_DIR, output_name)
    os.makedirs(output_dir, exist_ok=True)
    json_files = glob.glob(os.path.join(JSON_DIR, output_name, '*.json'))
    print(json_files)
        
    for json_file in json_files:
        print(json_file)
        if not os.path.exists(os.path.join(output_dir,os.path.basename(json_file))):
            json_path = json_file
            model_dir = MODEL_DIR
            db_dir = PUBLIC_DATA_DIR
            gpu_device = gpu_id
            output_dir = output_dir
            os.system(f"python alphafold3/run_alphafold.py --json_path={json_path} --model_dir={model_dir} --db_dir={db_dir} --gpu_device={gpu_device} --output_dir={output_dir}")
