import pandas as pd
from Bio.Seq import Seq
import os

df=pd.read_csv("./output/141409_7.tsv",sep="\t")
df.to_csv("./output/141409_7.csv",sep=",")

# Set your file path
csv_path = "./output/141409_7.tsv"  # Change this to your file as needed

# Extract the sample name from the file path (without extension)
sample_name = os.path.basename(csv_path).replace('.tsv', '')

# Extract the barcode (everything before first '_contig_')
df['barcode'] = df['sequence_id'].str.replace(r'_contig_.*$', '', regex=True)
print(df[['sequence_id', 'barcode']].head())
# Remove gaps/dots and translate
def translate_alignment(nt):
    if pd.isnull(nt):
        return ""
    nt_clean = nt.replace(".", "").replace("-", "")  # Remove IMGT dots and dashes
    nt_clean = nt_clean.upper().replace(' ', '').replace('\n','')
    # Optionally, ensure length is a multiple of 3 (trim excess)
    nt_clean = nt_clean[:len(nt_clean) - len(nt_clean) % 3]
    try:
        aa = str(Seq(nt_clean).translate())
        return aa
    except Exception as e:
        return ""

# Create amino acid column
df['sequence_aa'] = df['sequence_alignment'].apply(translate_alignment)
# Add the sample column
df['sample'] = sample_name
# Subset heavy and light chains
df_H = df[df['locus'] == 'IGH'].copy()
df_L = df[df['locus'].isin(['IGK','IGL'])].copy()
# Rename columns for merging
df_H = df_H.rename(columns={'sequence_alignment':'H_nt', 'sequence_aa':'H_aa'})
df_L = df_L.rename(columns={'sequence_alignment':'L_nt', 'sequence_aa':'L_aa'})
# Keep only relevant columns and barcode
df_H = df_H[['barcode', 'H_nt', 'H_aa', 'c_call','clone_id','sample']]
df_L = df_L[['barcode', 'L_nt', 'L_aa', 'c_call','clone_id','sample']]
# Pair heavy and light by barcode (inner join)
paired = pd.merge(df_H, df_L, on='barcode', how='inner')
# Save paired result to TSV
paired.to_csv("paired_HL141409_7.tsv", sep="\t", index=False)