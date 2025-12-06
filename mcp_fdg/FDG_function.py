##ddg的包
import os
import sys
import argparse
import torch
import pandas as pd
from ddg_models.predictor import DDGPredictor
from ddg_utils.misc import *
from ddg_utils.data import *
from ddg_utils.protein import *

##gearbind的包
import os
import sys
import math
import pprint
import pickle
import pandas as pd
from tqdm import tqdm
import torch
import numpy as np
from torchdrug.utils import comm
#这个需要跑两次
try:
    from torchdrug import core, models, data, utils
except:
    from torchdrug import core, models, data, utils
import types
from easydict import EasyDict as edict
from gearbind import dataset, layer, model, task, util
import csv
import glob
from Bio.PDB import PDBParser
from torchdrug.core import Registry as R
from torch.utils import data as torch_data
from torch.nn import functional as F
from gearbind import residue_constants
import shutil
import warnings
from Bio import PDB,SeqIO
from collections import defaultdict
#Foldx
import os
import pandas as pd
import shutil  # shutil模块用于文件操作



##ddg
ckpt = torch.load("ddg_checkpoints/model.pt")
config = ckpt['config']
weight = ckpt['model']
DDGmodel = DDGPredictor(config.model).to('cuda')
DDGmodel.load_state_dict(weight)
def ddg_predict(wt_pdb,mut_pdb):
    batch = load_wt_mut_pdb_pair(wt_pdb,mut_pdb)
    batch = recursive_to(batch, 'cuda')
    with torch.no_grad():
        DDGmodel.eval()
        pred = DDGmodel(batch['wt'], batch['mut'])
    return float(pred.cpu())

##gearbind

cfg = { 'task': {'class': 'BindingAffinityChange',
  'model': {'class': 'BindModel',
   'num_mlp_layer': 2,
   'model': {'class': 'GearBind',
    'input_dim': 58,
    'hidden_dims': [128, 128, 128, 128],
    'batch_norm': True,
    'short_cut': True,
    'concat_hidden': True,
    'num_relation': 7,
    'edge_input_dim': 59,
    'num_angle_bin': 8}},
  'graph_construction_model': {'class': 'GraphConstruction',
   'node_layers': [{'class': 'InterfaceGraph', 'cutoff': 6.0}],
   'edge_layers': [{'class': 'SequentialEdge', 'max_distance': 2},
    {'class': 'SpatialEdge', 'radius': 10.0, 'max_distance': 5},
    {'class': 'KNNEdge', 'k': 10, 'max_distance': 5}],
   'edge_feature': 'gearnet'},
  'normalization': False,
  'task': ['ddG'],
  'criterion': 'mse',
  'metric': ['mae', 'rmse', 'spearmanr', 'pearsonr']},
 'optimizer': {'class': 'Adam', 'lr': 0.0001},
 'engine': {'gpus': [0], 'batch_size': 2},
 'checkpoints': ['./gearbind_checkpoints/cl_gearbind0.pth',
                './gearbind_checkpoints/cl_gearbind1.pth',
                './gearbind_checkpoints/cl_gearbind2.pth',
                './gearbind_checkpoints/cl_gearbind3.pth',
                './gearbind_checkpoints/cl_gearbind4.pth',]}


def dump(cfg, dataset, solver):
    dataloader = data.DataLoader(dataset, solver.batch_size, shuffle=False, num_workers=0)
    device = torch.device(solver.gpus[0])
    solver.model.eval()
    preds = []
    for batch in dataloader:
        batch = utils.cuda(batch, device=device)
        with torch.no_grad():
            output = solver.model.predict(batch)
            preds.append(output.detach().cpu().numpy())
    pred = np.concatenate(preds, axis=0)
    return pred



def bio_load_pdb(pdb):
    parser = PDBParser(QUIET=True)
    protein = parser.get_structure(0, pdb)
    residues = [residue for residue in protein.get_residues()]
    residue_type = [data.Protein.residue2id.get(residue.get_resname(), 0) for residue in residues]
    chain_id = [data.Protein.alphabet2id.get(residue.get_parent().id, 0) for residue in residues]
    insertion_code = [data.Protein.alphabet2id.get(residue.full_id[3][2], -1) for residue in residues]
    residue_number = [residue.full_id[3][1] for residue in residues]
    id2residue = {residue.full_id: i for i, residue in enumerate(residues)}
    # residue_feature = functional.one_hot(torch.as_tensor(residue_type), len(data.Protein.residue2id)+1)

    atoms = [atom for atom in protein.get_atoms()]
    atoms = [atom for atom in atoms if atom.get_name() in data.Protein.atom_name2id]
    occupancy = [atom.get_occupancy() for atom in atoms]
    b_factor = [atom.get_bfactor() for atom in atoms]
    atom_type = [data.feature.atom_vocab.get(atom.get_name()[0], 0) for atom in atoms]
    atom_name = [data.Protein.atom_name2id.get(atom.get_name(), 37) for atom in atoms]
    node_position = np.stack([atom.get_coord() for atom in atoms], axis=0)
    node_position = torch.as_tensor(node_position)
    atom2residue = [id2residue[atom.get_parent().full_id] for atom in atoms]

    edge_list = [[0, 0, 0]]
    bond_type = [0]

    return data.Protein(edge_list, atom_type=atom_type, bond_type=bond_type, residue_type=residue_type,
                num_node=len(atoms), num_residue=len(residues), atom_name=atom_name,
                atom2residue=atom2residue, occupancy=occupancy, b_factor=b_factor, chain_id=chain_id,
                residue_number=residue_number, node_position=node_position, insertion_code=insertion_code, # residue_feature=residue_feature
            ), "".join([data.Protein.id2residue_symbol[res] for res in residue_type])



class SKEMPI(data.ProteinDataset):

    #fname = "SKEMPI.zip"
    md5 = "2c54e2ae7cda20cc5dfb2f5ab2adb8af"
    processed_file = "skempi.pkl.gz"
    splits = ["split_0", "split_1", "split_2", "split_3", "split_4"]

    def __init__(self, path, verbose=1, **kwargs):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        zip_file = os.path.join(path, self.fname)
        path = os.path.join(utils.extract(zip_file), "SKEMPI")
        #pkl_file = os.path.join(path, self.processed_file)

        # if os.path.exists(pkl_file):
        #     self.load_pickle(pkl_file, verbose=verbose, **kwargs)
        # else:
        pdb_files = []
        csv_files = []
        for split in self.splits:
            split_path = utils.extract(os.path.join(path, "%s.zip" % split))
            pdb_files += sorted(glob.glob(os.path.join(split_path, split, "*.pdb")))
            csv_files.append(os.path.join(path, "%s.csv" % split))
        self.load_pdbs(pdb_files, verbose=verbose, **kwargs)
        self.load_annotation(csv_files)
        #self.save_pickle(pkl_file, verbose=verbose)

        pdb_splits = [os.path.basename(os.path.dirname(pdb_file)) for pdb_file in self.pdb_files]
        self.num_samples = [pdb_splits.count(split) for split in self.splits]

    def load_pdbs(self, pdb_files, transform=None, lazy=False, verbose=0, **kwargs):
        """
        Load the dataset from pdb files.

        Parameters:
            pdb_files (list of str): pdb file names
            transform (Callable, optional): protein sequence transformation function
            lazy (bool, optional): if lazy mode is used, the proteins are processed in the dataloader.
                This may slow down the data loading process, but save a lot of CPU memory and dataset loading time.
            verbose (int, optional): output verbose level
            **kwargs
        """
        num_sample = len(pdb_files)

        self.transform = transform
        self.lazy = lazy
        self.kwargs = kwargs
        self.data = []
        self.pdb_files = []
        self.sequences = []

        if verbose:
            pdb_files = tqdm(pdb_files, "Constructing proteins from pdbs")
        for i, pdb_file in enumerate(pdb_files):
            if not lazy or i == 0:
                protein, sequence = bio_load_pdb(pdb_file)
            else:
                protein, sequence = None, None
            self.data.append(protein)
            self.pdb_files.append(pdb_file)
            self.sequences.append(sequence)

    def load_annotation(self, csv_files):
        data_dict = {
            os.path.basename(pdb_file): (protein, pdb_file, sequence) \
                for pdb_file, protein, sequence in zip(self.pdb_files, self.data, self.sequences)
        }
        self.data = []
        self.pdb_files = []
        self.sequences = []

        for fname in csv_files:
            csv_file = open(fname, "r")
            reader = csv.reader(csv_file, delimiter=',')
            header = next(reader)
            mutation_id, chain_a_id, chain_b_id, wt_protein_id, mt_protein_id = \
                map(header.index, ["mutation", "chain_a", "chain_b", "wt_protein", "mt_protein"])
            ddG_id = header.index("ddG") if "ddG" in header else None

            for line in reader:
                mutations, chain_a, chain_b, _wild_type, _mutant = \
                    map(lambda i: line[i], [mutation_id, chain_a_id, chain_b_id, wt_protein_id, mt_protein_id])
                ddG = line[ddG_id] if ddG_id is not None else 0.0
                mutations = mutations.split(",")

                if _wild_type not in data_dict: continue
                wild_type = data_dict[_wild_type][0]
                with wild_type.node():
                    entity_a = torch.zeros(wild_type.num_residue, dtype=torch.bool)
                    for a in chain_a:
                        entity_a |= wild_type.chain_id == wild_type.alphabet2id[a]
                    wild_type.entity_a = entity_a[wild_type.atom2residue]

                    entity_b = torch.zeros(wild_type.num_residue, dtype=torch.bool)
                    for b in chain_b:
                        entity_b |= wild_type.chain_id == wild_type.alphabet2id[b]
                    wild_type.entity_b = entity_b[wild_type.atom2residue]

                    is_mutation = torch.zeros(wild_type.num_residue, dtype=torch.bool)
                    for m in mutations:
                        if m[-2].isalpha():
                            is_mutation |= \
                                (wild_type.chain_id == wild_type.alphabet2id[m[1]]) & \
                                (wild_type.residue_number == int(m[2:-2])) & \
                                (wild_type.insertion_code == wild_type.alphabet2id[m[-2]])
                        else:
                            is_mutation |= \
                                (wild_type.chain_id == wild_type.alphabet2id[m[1]]) & \
                                (wild_type.residue_number == int(m[2:-1]))
                    wild_type.is_mutation = is_mutation[wild_type.atom2residue]
                wild_type = wild_type.subgraph(wild_type.entity_a | wild_type.entity_b)
                if hasattr(wild_type, "node_feature"):
                    with wild_type.node():
                        wild_type.node_feature = wild_type.node_feature.to_sparse()

                if _mutant not in data_dict: continue
                mutant = data_dict[_mutant][0]
                with mutant.node():
                    entity_a = torch.zeros(mutant.num_residue, dtype=torch.bool)
                    for a in chain_a:
                        entity_a |= mutant.chain_id == mutant.alphabet2id[a]
                    mutant.entity_a = entity_a[mutant.atom2residue]

                    entity_b = torch.zeros(mutant.num_residue, dtype=torch.bool)
                    for b in chain_b:
                        entity_b |= mutant.chain_id == mutant.alphabet2id[b]
                    mutant.entity_b = entity_b[mutant.atom2residue]

                    is_mutation = torch.zeros(mutant.num_residue, dtype=torch.bool)
                    for m in mutations:
                        if m[-2].isalpha():
                            is_mutation |= \
                                (mutant.chain_id == mutant.alphabet2id[m[1]]) & \
                                (mutant.residue_number == int(m[2:-2])) & \
                                (mutant.insertion_code == mutant.alphabet2id[m[-2]])
                        else:
                            is_mutation |= \
                                (mutant.chain_id == mutant.alphabet2id[m[1]]) & \
                                (mutant.residue_number == int(m[2:-1]))
                    mutant.is_mutation = is_mutation[mutant.atom2residue]
                mutant = mutant.subgraph(mutant.entity_a | mutant.entity_b)
                if hasattr(mutant, "node_feature"):
                    with mutant.node():
                        mutant.node_feature = mutant.node_feature.to_sparse()

                self.data.append((wild_type, mutant, float(ddG), mutations, fname.split(".")[0]))
                self.pdb_files.append(data_dict[_mutant][1])
                self.sequences.append((data_dict[_wild_type][2], data_dict[_mutant][2]))

    def split(self, test_set="split_0", valid_ratio=0.1):
        indices = list(range(len(self)))
        train_indices = []
        offset = 0
        for split, num_samples in zip(self.splits, self.num_samples):
            if split != test_set:
                train_indices += indices[offset: offset + num_samples]
            offset += num_samples

        idx = self.splits.index(test_set)
        num_samples = self.num_samples[idx]
        offset = sum(self.num_samples[:idx])
        test_indices = indices[offset: offset + num_samples]

        num_val_samples = int(len(train_indices) * valid_ratio)
        valid_indices = np.random.choice(train_indices, num_val_samples, replace=False)
        train_indices = [idx for idx in train_indices if idx not in valid_indices]

        return [
            torch_data.Subset(self, train_indices),
            torch_data.Subset(self, valid_indices),
            torch_data.Subset(self, test_indices)
        ]

    def get_item(self, index):
        if getattr(self, "lazy", False):
            mutant = data.Protein.from_pdb(self.pdb_files[index], self.kwargs)
            wild_type = data.Protein.from_pdb(
                os.path.join(os.path.dirname(self.pdb_files[index]), "WT_" + os.path.basename(self.pdb_files[index])),
                self.kwargs
            )
        else:
            wild_type = self.data[index][0].clone()
            mutant = self.data[index][1].clone()

        wt_residue_feature = F.one_hot(wild_type.residue_type, len(data.Protein.residue2id)+1)
        # wt_atom_feature = F.one_hot(wild_type.atom_name, len(data.Protein.atom_name2id)+1)
        wt_atom_feature = torch.cat([
            F.one_hot(wild_type.atom_name, residue_constants.atom_type_num),
            wt_residue_feature[wild_type.atom2residue]
        ], dim=-1)
        with wild_type.node():
            wild_type.node_feature = wt_atom_feature
        with wild_type.residue():
            wild_type.residue_feature = wt_residue_feature

        mt_residue_feature = F.one_hot(mutant.residue_type, len(data.Protein.residue2id)+1)
        # mt_atom_feature = F.one_hot(mutant.atom_name, len(data.Protein.atom_name2id)+1)
        mt_atom_feature = torch.cat([
            F.one_hot(mutant.atom_name, residue_constants.atom_type_num),
            mt_residue_feature[mutant.atom2residue]
        ], dim=-1)
        with mutant.node():
            mutant.node_feature = mt_atom_feature
        with mutant.residue():
            mutant.residue_feature = mt_residue_feature
        # if hasattr(wild_type, "node_feature"):
        #     with wild_type.node():
        #         wild_type.node_feature = wild_type.node_feature.to_dense()
        # if hasattr(wild_type, "residue_feature"):
        #     with wild_type.residue():
        #         wild_type.residue_feature = wild_type.residue_feature.to_dense()
        # if hasattr(mutant, "node_feature"):
        #     with mutant.node():
        #         mutant.node_feature = mutant.node_feature.to_dense()
        # if hasattr(mutant, "residue_feature"):
        #     with mutant.residue():
        #         mutant.residue_feature = mutant.residue_feature.to_dense()
        item = {"wild_type": wild_type, "mutant": mutant}
        if self.transform:
            item = self.transform(item)
        item["ddG"] = self.data[index][2]
        return item

    def __repr__(self):
        lines = [
            "#sample: %d" % len(self),
            "#task: ddG",
        ]
        return "%s(\n  %s\n)" % (self.__class__.__name__, "\n  ".join(lines))

# @R.register("test_data")
class test_data(SKEMPI):

    #processed_file = "NK138.pkl.gz"
    #splits = os.listdir("input")#["test"]

    def __init__(self, path,split_list, verbose=1, **kwargs):
        self.splits = split_list
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        #pkl_file = os.path.join(path, self.processed_file)

        # if os.path.exists(pkl_file):
        #     self.load_pickle(pkl_file, verbose=verbose, **kwargs)
        # else:
        pdb_files = []
        csv_files = []
        #print("splits:",self.splits)
        for split in self.splits:
            split_path = os.path.join(path, split)
            #print("split_path",split_path)
            pdb_files += sorted(glob.glob(os.path.join(split_path, "data", "*.pdb")))
            csv_files.append(os.path.join(split_path, "data.csv"))
        #print("pdb_files:",pdb_files)
        #print(pdb_files)
        #print(pdb_files)
        #print(csv_files)
        self.load_pdbs(pdb_files, verbose=verbose, **kwargs)
        self.load_annotation(csv_files)
        pdb_files = []
        for pdb_file in self.pdb_files:
            pdb_dir, pdb_name = os.path.split(pdb_file)
            split = os.path.basename(os.path.dirname(pdb_dir))
            pdb_file = os.path.join(split, pdb_name)
            pdb_files.append(pdb_file)
        self.pdb_files = pdb_files
        #self.save_pickle(pkl_file, verbose=verbose)

        pdb_splits = [os.path.basename(os.path.dirname(pdb_file)) for pdb_file in self.pdb_files]
        self.num_samples = [pdb_splits.count(split) for split in self.splits]

    def split(self, test_set="1n8z_renum.pdb_HL_C"):
        indices = list(range(len(self)))
        test_indices = []
        offset = 0
        for split, num_samples in zip(self.splits, self.num_samples):
            if split == test_set:
                test_indices += indices[offset: offset + num_samples]
            offset += num_samples

        return [
            torch_data.Subset(self, test_indices),
            torch_data.Subset(self, test_indices),
            torch_data.Subset(self, test_indices)
        ]


one_letter ={'VAL':'V', 'ILE':'I', 'LEU':'L', 'GLU':'E', 'GLN':'Q', \
'ASP':'D', 'ASN':'N', 'HIS':'H', 'TRP':'W', 'PHE':'F', 'TYR':'Y',    \
'ARG':'R', 'LYS':'K', 'SER':'S', 'THR':'T', 'MET':'M', 'ALA':'A',    \
'GLY':'G', 'PRO':'P', 'CYS':'C'}
def get_sequence(parser,pdb_path):
    structure = parser.get_structure("temp",pdb_path)
    x=list([str(i)[-2:-1] for i in structure.get_chains()])
    #print(x)
    if "H" not in x:
        print("cant find H chain!!")
        return False
    #print(x)
    chain_H=list(structure.get_chains())[x.index("H")]
    residues_H=list(chain_H.get_residues())
    sequence=""
    X_num=0
    for ind,res in enumerate(residues_H):
        #print(res)
        cur=res.id[1]
        sequence+=one_letter[res.get_resname()]
    return sequence

def get_chain(parser,pdb_path):
    structure = parser.get_structure("temp",pdb_path)
    x=list([str(i)[-2:-1] for i in structure.get_chains()])
    return x

final_dataset= {'class': 'test_data',
  'path': './temp/',
  'node_feature': 'residue_symbol',
  'residue_feature': 'default',
  'split': {'test_set': "gearbind"}
               }
cfg["dataset"]=final_dataset
cfg=edict(cfg)
solvers=[]
dataset = test_data(path=cfg.dataset.path, split_list=[cfg.dataset.split.test_set])
for i in range(len(cfg.checkpoints)): #直接存储模型
    cfg.checkpoint = cfg.checkpoints[i]
    solvers.append(util.build_solver(cfg, dataset))

def test(cfg,dataset):
    preds = []
    #得加载数据
    dataset = test_data(path=cfg.dataset.path, split_list=[cfg.dataset.split.test_set])
    for i in range(len(cfg.checkpoints)):
        pred = dump(cfg, dataset, solvers[i])
        preds.append(pred)
    pred = np.stack(preds, axis=0)
    return pred

def gearbind_predict(wt_pdb, mut_pdb):  # 代表重链对应的位置
    parser = PDB.PDBParser()
    WT_H = get_sequence(parser, wt_pdb)
    MT_H = get_sequence(parser, mut_pdb)
    chains = get_chain(parser, wt_pdb)
    # print(WT_H,MT_H)
    mutations = []
    for j in range(len(WT_H)):
        if WT_H[j] != MT_H[j]:
            # print()
            mutations.append(f"{WT_H[j]}H{j + 1}{MT_H[j]}")
    mutations = ",".join(mutations)
    D = {}
    D["pdb_id"] = ["test_data"]
    D["mutation"] = [mutations]
    D["chain_a"] = ["".join([i for i in chains if i in ["H", "L"]])]
    D["chain_b"] = ["".join([i for i in chains if i not in ["H", "L"]])]
    D["wt_protein"] = ["WT.pdb"]
    D["mt_protein"] = ["MUT.pdb"]
    pd.DataFrame(D).to_csv("temp/gearbind/data.csv")

    shutil.copy(wt_pdb, f"temp/gearbind/data/WT.pdb")
    shutil.copy(mut_pdb, f"temp/gearbind/data/MUT.pdb")
    return test(cfg,dataset)

def foldx_repair(before_repair_path,after_repair_path):
    output_name = after_repair_path  # 新建的文件夹名称
    if os.path.exists(output_name):
        shutil.rmtree(output_name)  # 删除整个文件夹
        print(f"文件夹 '{output_name}' 已删除。")
    os.makedirs(output_name)
    print(f"文件夹 '{output_name}' 已重新创建。")
    #移入foldx
    shutil.copy(f"./foldx_bin/foldx", f"{after_repair_path}/foldx")
    shutil.copy(f"./foldx_bin/rotabase.txt", f"{after_repair_path}/rotabase.txt")
    #进行修复
    for br_pdb in os.listdir(before_repair_path): #对应的pdb
        if ".pdb" not in br_pdb:continue
        shutil.copy(f"{before_repair_path}/{br_pdb}",f"{after_repair_path}/{br_pdb}")
        os.system(f"cd {after_repair_path} && ./foldx --command=RepairPDB --pdb={br_pdb}")
    return
    #break



def foldx_helper(wt_path,mutations,is_repair=True):
    output_name = "./temp/foldx"  # 新建的文件夹名称
    if os.path.exists(output_name):
        shutil.rmtree(output_name)  # 删除整个文件夹
        print(f"文件夹 '{output_name}' 已删除。")
    os.makedirs(output_name)
    print(f"文件夹 '{output_name}' 已重新创建。")

    #每次操作将foldx复制到指定文件夹
    shutil.copy(wt_path, f"temp/foldx/origin.pdb")
    shutil.copy(f"./foldx_bin/foldx", f"temp/foldx/foldx")
    shutil.copy(f"./foldx_bin/rotabase.txt", f"temp/foldx/rotabase.txt")
    with open(f"temp/foldx/individual_list.txt", 'w', encoding='utf-8') as file:
        for item in mutations:
            file.write(item + ";\n")
    #然后进入指定文件夹并运行foldx操作
    #先进行修复
    #break
    if (is_repair):
        os.system("cd temp/foldx && ./foldx --command=RepairPDB --pdb=origin.pdb")
        os.rename("temp/foldx/origin_repair.pdb","temp/foldx/origin.pdb")
    os.system("cd temp/foldx && ./foldx --command=BuildModel --pdb=origin.pdb --mutant-file=individual_list.txt --numberOfRuns=1")
    #最后把表格保存起来
    lines_list = []
    with open(f"temp/foldx/individual_list.txt", 'r', encoding='utf-8') as file:
        for line in file:
            lines_list.append(line.strip()[:-1])
    file_path=f"temp/foldx/Dif_origin.fxout"
    foldx_data=pd.read_csv(file_path, sep="	", skiprows=8, engine='python')[["Pdb","total energy"]].rename(columns={"total energy":"build-model_energy"})
    foldx_data["mutant"]=lines_list
    return foldx_data
    #break

def main():
    """
    主函数，执行程序的主要逻辑。
    """
    wt_path = "./input/example/WT_INF-26_Repair_1.pdb"
    mut_path = "./input/example/INF-26_Repair_1.pdb"
    mutations = ["YH75D", "YH75Q,QH1E"]
    # 检查命令行参数
    ddg_pred = ddg_predict(wt_path, mut_path)
    gearbind_pred = gearbind_predict(wt_path, mut_path)
    foldx_Data=foldx_helper(wt_path, mutations, is_repair=False)
    # 代表一个WT_path和一组突变

    print("ddg_pred:",ddg_pred)
    print("gearbind_pred:",gearbind_pred)
    print("[shape]foldx_Data:",foldx_Data.shape)


if __name__ == "__main__":
    main()