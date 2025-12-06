import os
import re
import torch
import random
import numpy as np
import pandas as pd
from torch.utils.data import random_split
from transformers import BertTokenizer
from torch.utils.data import Dataset as Dataset_n
from torch.utils.data import DataLoader as DataLoader_n
from torch.utils.data import WeightedRandomSampler

# from torch_geometric.loader import DataLoader as DataLoader_n
# from torch_geometric.data import Data


random.seed(339)
try:
    tokenizer = BertTokenizer.from_pretrained("./prot_bert", do_lower_case=False)
except:
    tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
len_a = len(tokenizer.get_vocab().keys())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

max_seq_antib = 250
max_seq_antib_concat = 400
# max_seq_antib = 150
max_seq_antig = 400  # XBCR-net
max_seq_antig_hiv = 1000
max_seq_antig_rbd = 300
max_seq_antig_flu = 700


ALPHABET = "ACDEFGHIKLMNPQRSTVWY"

def seq_filt(s,STR_REP=''):
    s = s.replace(' ', STR_REP)
    s = s.replace('\n', STR_REP)
    s = s.replace('\t', STR_REP)
    s = s.replace('_', STR_REP)
    s = s.replace('|', STR_REP)
    return s

def one_hot_encoder(s, alphabet=ALPHABET, random_disturb_scale=0.0):
    """
    One hot encoding of a biological sequence.

    Parameters
    ---
    s: str, sequence which should be encoded
    alphabet: Alphabet object, downloaded from
        http://biopython.org/DIST/docs/api/Bio.Alphabet.IUPAC-module.html

    Example
    ---
    sequence = 'CARGSSYSSFAYW'
    one_hot_encoder(s=sequence, alphabet=IUPAC.protein)

    Returns
    ---
    x: array, n_size_alphabet, n_length_string
        Sequence as one-hot encoding
    """


    # Build dictionary
    # d = {a: i for i, a in enumerate(alphabet.letters)}
    d = {a: i for i, a in enumerate(alphabet)}

    # Encode
    x = np.zeros((len(s), len(d) + 1))
    # x[range(len(s)),[d[c] if c in alphabet.letters else len(d) for c in s]] = 1
    x[range(len(s)), [d[c] if c in alphabet else len(d) for c in s]] = 1
    # if any(x[:,len(d)]>0):
    #     print(s)

    # disturbance
    if random_disturb_scale > 0.:
        x += np.random.uniform(0, random_disturb_scale, x.shape)
        x = x / np.sum(x, axis=-1, keepdims=True)
    return x[:, :len(d)]


def vec_shift(vec, seq_shift=20, seq_len=150):
    vec_tmp = np.zeros_like(vec)
    vec_tmp[seq_shift:seq_shift + seq_len, ...] = vec[:seq_len, ...]
    return vec_tmp


class Ab_Dataset(Dataset_n):
    def __init__(self, datalist, proportions=None, sample_func=None,
                 n_samples=0, is_rand_sample=True, onehot=False, rand_shift=True):
        print('AB_Dataset initializing...')
        assert type(datalist) == list, 'Expect [datalist] to be <list> type but got <{}>.'.format(type(datalist))
        assert type(proportions) == list, 'Expect [proportions] to be <list> type but got <{}>.'.format(
            type(proportions))
        assert type(sample_func) == list, 'Expect [sample_func] to be <list> type but got <{}>.'.format(
            type(sample_func))
        assert len(datalist) == len(proportions) == len(
            sample_func), '[datalist], [proportions] and [sample_func] should have same length, however got ' \
                          'len(datalist)={}, len(proportions)={}, and len(sample_func)={}'.format(len(datalist),
                                                                                                  len(proportions),
                                                                                                  len(sample_func))
        assert n_samples > 0, 'Num of samples should be a positive number instead of {}'.format(n_samples)

        self.datalist = datalist
        self.proportions = proportions
        self.prop_norm()
        self.is_rand_sample = is_rand_sample
        self.n_samples = n_samples
        self.onehot = onehot
        self.rand_shift = rand_shift

        self.ag_head_str = "variant_seq"
        self.label_str = 'rbd'
        self.sample_func = sample_func
        self.sample_func_dict = {'sample': self.sample, 'rand_sample': self.rand_sample,
                                 'rand_sample_rand_combine': self.rand_sample_rand_combine,
                                 'rand_sample_cross_eval': self.rand_sample_cross_eval}

        # if self.mode == "train":
        #     self.sars_exp = pd.concat([self.sars_pos.copy(), self.sars_neg.copy()], ignore_index=True)

    def __len__(self):
        return self.n_samples

    def prop_norm(self):
        x = np.array(self.proportions, dtype=float)
        x /= np.sum(x)
        y = []
        for i in range(x.shape[0]):
            y.append(np.sum(x[:(i + 1)]))
        self.proportions = y
        print('proportions has been normalized to {}'.format(self.proportions))

    def sample(self, data, idx):
        return data.loc[idx].copy()

    def rand_sample(self, data, idx):
        rand_index = random.randint(0, data.shape[0] - 1)
        return data.loc[rand_index].copy()

    def rand_sample_rand_combine(self, data, idx):
        data1, data2 = data
        # this is for non-experiment data
        # fetch antigen from experiment data
        rand_index = random.randint(0, data1.shape[0] - 1)
        d = data1.loc[rand_index].copy()
        # fetch heavy/light chain from non-experiment data
        for ii in range(1000):
            rand_index = random.randint(0, data2.shape[0] - 1)
            d['Heavy'] = data2.loc[rand_index, 'Heavy']
            if type(d['Heavy']) != float:
                break
        #
        for ii in range(1000):
            rand_index = random.randint(0, data2.shape[0] - 1)
            d['Light'] = data2.loc[rand_index, 'Light']
            if type(d['Light']) != float:
                break
        #
        d[self.label_str] = 0
        return d

    def rand_sample_cross_eval(self, data, idx):
        data1, data2 = data
        n_rand = random.random()
        #
        rand_index = random.randint(0, data1.shape[0] - 1)
        d = data1.loc[rand_index].copy()
        #
        rand_index = random.randint(0, data2.shape[0] - 1)
        if n_rand < 0.5:
            d[self.ag_head_str] = data2.loc[rand_index, self.ag_head_str]
        else:
            d['Heavy'] = data2.loc[rand_index, 'Heavy']
            d['Light'] = data2.loc[rand_index, 'Light']
        d[self.label_str] = 0
        return d

    def __getitem__(self, index):
        if self.is_rand_sample:
            num_rand = random.random()
            for i, _prop in enumerate(self.proportions):
                if num_rand <= _prop:
                    data = self.sample_func_dict[self.sample_func[i]](self.datalist[i], index)
                    break
        else:

            data = self.sample_func_dict[self.sample_func[0]](self.datalist[0], index)

        # if self.mode == "train":
        #     num_rand = random.random()
        #     if num_rand < 0.2:
        #         data = self.rand_sample(self.sars.pos)
        #     elif num_rand < 0.8:
        #         data = self.rand_sample(self.sars.neg)
        #     else:
        #         data = self.rand_sample_rand_combine(self.sars_exp, self.sars_nonexp)
        # elif self.mode == "meta":
        #     num_rand = random.random()
        #     if num_rand < 0.25:
        #         data = self.rand_sample(self.sars_pos)
        #     else:
        #         data = self.rand_sample(self.sars_neg)
        # else:
        #     data = self.sample(self.sars_pos, index)

        # Capitalize
        seq_ab_v = data['Heavy'].upper()
        seq_ab_l = data['Light'].upper()

        seq_ag = data[self.ag_head_str].upper()

        # Replace illegal strings
        seq_ab_v = re.sub(r'[UZOB*_]', "X", seq_ab_v)
        seq_ab_l = re.sub(r'[UZOB*_]', "X", seq_ab_l)

        seq_ag = re.sub(r'[UZOB*_]', "X", seq_ag)

        # seq_ab_v = seq_filt(seq_ab_v)
        # seq_ab_l = seq_filt(seq_ab_l)
        # seq_ag = seq_filt(seq_ag)

        ids_ag = np.zeros([max_seq_antig, 20])
        # ids_ag = np.zeros([800, 30])
        ids_ag[0:0 + len(seq_ag), :] = one_hot_encoder(s=seq_ag)

        # ids_ag[5:5 + len(onehot_encoded), :] = np.array(onehot_encoded)
        input_ids_ag = ids_ag

        # index
        # Put a space between every two letters
        ids_ab_v = tokenizer(" ".join(list(seq_ab_v)), return_tensors='pt', max_length=max_seq_antib,
                             padding='max_length',
                             truncation=True)
        ids_ab_l = tokenizer(" ".join(list(seq_ab_l)), return_tensors='pt', max_length=max_seq_antib,
                             padding='max_length',
                             truncation=True)

        # input_ids_ab_v = torch.tensor(ids_ab_v['input_ids'])
        # input_ids_ab_l = torch.tensor(ids_ab_l['input_ids'])
        input_ids_ab_v = ids_ab_v['input_ids']
        input_ids_ab_l = ids_ab_l['input_ids']
        input_ids_ab_v = input_ids_ab_v.squeeze(0)
        input_ids_ab_l = input_ids_ab_l.squeeze(0)

        # attention_mask_ab_v = torch.tensor(ids_ab_v['attention_mask'])
        # attention_mask_ab_l = torch.tensor(ids_ab_l['attention_mask'])
        attention_mask_ab_v = ids_ab_v['attention_mask']
        attention_mask_ab_l = ids_ab_l['attention_mask']
        attention_mask_ab_v = attention_mask_ab_v.squeeze(0)
        attention_mask_ab_l = attention_mask_ab_l.squeeze(0)

        # # one-hot
        if self.onehot:
            ids_ab_v = np.zeros([max_seq_antib, 20], dtype=float)
            ids_ab_l = np.zeros([max_seq_antib, 20], dtype=float)
            ids_ab_v[0:0 + len(seq_ab_v), :] = one_hot_encoder(s=seq_ab_v)
            ids_ab_l[0:0 + len(seq_ab_l), :] = one_hot_encoder(s=seq_ab_l)
            input_ids_ab_v = ids_ab_v
            input_ids_ab_l = ids_ab_l

            # print(np.shape(input_ids_ab_l))
        if self.rand_shift:
            shift_ab_v = np.random.randint(0, max_seq_antib - len(seq_ab_v))
            shift_ab_l = np.random.randint(0, max_seq_antib - len(seq_ab_l))
            shift_ag = np.random.randint(0, max_seq_antig - len(seq_ag))
        else:
            shift_ab_v = 15
            shift_ab_l = 15
            shift_ag = 15
        input_ids_ab_v = vec_shift(input_ids_ab_v, seq_shift=shift_ab_v, seq_len=len(seq_ab_v))
        attention_mask_ab_v = vec_shift(attention_mask_ab_v, seq_shift=shift_ab_v, seq_len=len(seq_ab_v))
        input_ids_ab_l = vec_shift(input_ids_ab_l, seq_shift=shift_ab_l, seq_len=len(seq_ab_l))
        attention_mask_ab_l = vec_shift(attention_mask_ab_l, seq_shift=shift_ab_l, seq_len=len(seq_ab_l))
        input_ids_ag = vec_shift(input_ids_ag, seq_shift=shift_ag, seq_len=len(seq_ag))
        # antigen_graph = 0

        # charge = data['charge']
        # hydro = data['hydro']
        # flu neu
        label=data[self.label_str]
        # try:
        #     if data[self.label_str] == 1:
        #         label = 1
        #     else:
        #         label = 0
        # except:
        #     label = -1
        #     print(label)
        # if data['value'] != 'n':
        #     value = float(data['value'])
        #     loss_ic50 = 1
        # else:
        #     value = 5000
        #     loss_ic50 = 0
        #
        # if data['Type'] == 'auto':
        #     auto = data[self.label_str]
        #     loss_auto = 1
        #     loss_main = 0
        # else:
        #     auto = 5000
        #     loss_auto = 0
        #     loss_main = 1

        loss_main = 1
        value, auto = 5000, 5000

        # print(seq_ag, seq_ab_v,seq_ab_l)
        # print(np.sum(input_ids_ag,axis=-1), np.sum(input_ids_ab_l,axis=-1))

        if self.onehot:
            return {'input_ids_ab_v': torch.tensor(input_ids_ab_v, dtype=torch.float32),
                    'attention_mask_ab_v': torch.tensor(attention_mask_ab_v, dtype=torch.float32),
                    'input_ids_ab_l': torch.tensor(input_ids_ab_l, dtype=torch.float32),
                    'attention_mask_ab_l': torch.tensor(attention_mask_ab_l, dtype=torch.float32),
                    'label': torch.tensor(label, dtype=torch.float32),
                    'input_ids_ag': torch.tensor(input_ids_ag, dtype=torch.float32),
                    'value': torch.tensor(value, dtype=torch.float32),
                    'auto': torch.tensor(auto, dtype=torch.float32),
                    'loss_main': torch.tensor(loss_main, dtype=torch.float32)}
            # ,torch.tensor(charge,dtype=torch.float32),torch.tensor(hydro,dtype=torch.float32)
        else:
            return {'input_ids_ab_v': torch.tensor(input_ids_ab_v),
                    'attention_mask_ab_v': torch.tensor(attention_mask_ab_v),
                    'input_ids_ab_l': torch.tensor(input_ids_ab_l),
                    'attention_mask_ab_l': torch.tensor(attention_mask_ab_l),
                    'label': torch.tensor(label, dtype=torch.float32),
                    'input_ids_ag': torch.tensor(input_ids_ag, dtype=torch.float32),
                    'value': torch.tensor(value, dtype=torch.float32),
                    'auto': torch.tensor(auto, dtype=torch.float32),
                    'loss_main': torch.tensor(loss_main, dtype=torch.float32)}
            # ,torch.tensor(charge,dtype=torch.float32),torch.tensor(hydro,dtype=torch.float32)


class Ab_Dataset_mean_teacher(Dataset_n):
    def __init__(self, datalist, proportions=None, sample_func=None,
                 n_samples=0, is_rand_sample=True, onehot=False, rand_shift=True):
        print('AB_Dataset initializing...')
        assert type(datalist) == list, 'Expect [datalist] to be <list> type but got <{}>.'.format(type(datalist))
        assert type(proportions) == list, 'Expect [proportions] to be <list> type but got <{}>.'.format(
            type(proportions))
        assert type(sample_func) == list, 'Expect [sample_func] to be <list> type but got <{}>.'.format(
            type(sample_func))
        assert len(datalist) == len(proportions) == len(
            sample_func), '[datalist], [proportions] and [sample_func] should have same length, however got ' \
                          'len(datalist)={}, len(proportions)={}, and len(sample_func)={}'.format(len(datalist),
                                                                                                  len(proportions),
                                                                                                  len(sample_func))
        assert n_samples > 0, 'Num of samples should be a positive number instead of {}'.format(n_samples)

        self.datalist = datalist
        self.proportions = proportions
        self.prop_norm()
        self.is_rand_sample = is_rand_sample
        self.n_samples = n_samples
        self.onehot = onehot
        self.rand_shift = rand_shift

        self.ag_head_str = "variant_seq"
        self.label_str = 'rbd'
        self.sample_func = sample_func
        self.sample_func_dict = {'sample': self.sample, 'rand_sample': self.rand_sample,
                                 'rand_sample_rand_combine': self.rand_sample_rand_combine,
                                 'rand_sample_cross_eval': self.rand_sample_cross_eval,
                                 'no_label': self.rand_sample_no_label}

        # if self.mode == "train":
        #     self.sars_exp = pd.concat([self.sars_pos.copy(), self.sars_neg.copy()], ignore_index=True)

    def __len__(self):
        return self.n_samples

    def prop_norm(self):
        x = np.array(self.proportions, dtype=float)
        x /= np.sum(x)
        y = []
        for i in range(x.shape[0]):
            y.append(np.sum(x[:(i + 1)]))
        self.proportions = y
        print('proportions has been normalized to {}'.format(self.proportions))

    def sample(self, data, idx):
        return data.loc[idx].copy()

    def rand_sample(self, data, idx):
        rand_index = random.randint(0, data.shape[0] - 1)
        return data.loc[rand_index].copy()

    def rand_sample_no_label(self, data, idx):
        data1, data2 = data

        rand_index = random.randint(0, data1.shape[0] - 1)
        d = data1.loc[rand_index].copy()
        d[self.label_str] = random.random()  # a random number from 0 to 1

        rand_index = random.randint(0, data2.shape[0] - 1)
        d[self.ag_head_str] = data2.loc[rand_index, self.ag_head_str]

        return d

    def rand_sample_rand_combine(self, data, idx):
        data1, data2 = data
        # this is for non-experiment data
        # fetch antigen from experiment data
        rand_index = random.randint(0, data1.shape[0] - 1)
        d = data1.loc[rand_index].copy()
        # fetch heavy/light chain from non-experiment data
        for ii in range(1000):
            rand_index = random.randint(0, data2.shape[0] - 1)
            d['Heavy'] = data2.loc[rand_index, 'Heavy']
            if type(d['Heavy']) != float:
                break
        #
        for ii in range(1000):
            rand_index = random.randint(0, data2.shape[0] - 1)
            d['Light'] = data2.loc[rand_index, 'Light']
            if type(d['Light']) != float:
                break
        #
        d[self.label_str] = 0
        return d

    def rand_sample_cross_eval(self, data, idx):
        data1, data2 = data
        n_rand = random.random()
        #
        rand_index = random.randint(0, data1.shape[0] - 1)
        d = data1.loc[rand_index].copy()
        #
        rand_index = random.randint(0, data2.shape[0] - 1)
        if n_rand < 0.5:
            d[self.ag_head_str] = data2.loc[rand_index, self.ag_head_str]
        else:
            d['Heavy'] = data2.loc[rand_index, 'Heavy']
            d['Light'] = data2.loc[rand_index, 'Light']
        d[self.label_str] = 0
        return d

    def random_maskoff_noise(self, data, length):
        off_mask = torch.ones_like(data)
        zero_num = int(random.uniform(0, 0.05) * length) + 1
        # zero_num = 0
        zero_indices = random.sample(range(length), zero_num)
        off_mask[zero_indices] = 0
        d = off_mask * data
        return d

    def seq_random_maskoff_noise(self, data):
        zero_num = int(random.uniform(1.01, 2.99))
        zero_indices = random.sample(range(10, len(data)-10), zero_num)
        d = list(data)
        for idx in zero_indices: d[idx] = 'X'
        return ''.join(d)

    def seq_random_flip(self, data):
        return data[::-1]

    def seq_random_prefix_suffix(self, data, max_len_prefix=3, max_len_suffix=3):
        num_prefix, num_suffix = int(random.uniform(0, max_len_prefix)), int(random.uniform(0, max_len_suffix))
        prefix = ''.join(random.choices(ALPHABET, k=num_prefix))
        suffix = ''.join(random.choices(ALPHABET, k=num_suffix))
        return prefix + data + suffix

    def seq_augmentation(self, data):
        if random.random() < 0.5: data = self.seq_random_maskoff_noise(data)  # random mask-off
        if random.random() < 0.5: data = self.seq_random_flip(data)  # random flip sequence
        if random.random() < 0.5: data = self.seq_random_prefix_suffix(data)
        return data

    def __getitem__(self, index):
        if self.is_rand_sample:
            num_rand = random.random()
            for i, _prop in enumerate(self.proportions):
                if num_rand <= _prop:
                    self.sample_mode = self.sample_func[i]
                    data = self.sample_func_dict[self.sample_func[i]](self.datalist[i], index)
                    break
        else:
            data = self.sample_func_dict[self.sample_func[0]](self.datalist[0], index)

        # if self.mode == "train":
        #     num_rand = random.random()
        #     if num_rand < 0.2:
        #         data = self.rand_sample(self.sars.pos)
        #     elif num_rand < 0.8:
        #         data = self.rand_sample(self.sars.neg)
        #     else:
        #         data = self.rand_sample_rand_combine(self.sars_exp, self.sars_nonexp)
        # elif self.mode == "meta":
        #     num_rand = random.random()
        #     if num_rand < 0.25:
        #         data = self.rand_sample(self.sars_pos)
        #     else:
        #         data = self.rand_sample(self.sars_neg)
        # else:
        #     data = self.sample(self.sars_pos, index)

        # Capitalize
        seq_ab_v = data['Heavy'].upper()
        seq_ab_l = data['Light'].upper()

        seq_ag = data[self.ag_head_str].upper()

        # Replace illegal strings
        seq_ab_v = re.sub(r'[UZOB*_]', "X", seq_ab_v)
        seq_ab_l = re.sub(r'[UZOB*_]', "X", seq_ab_l)

        seq_ag = re.sub(r'[UZOB*_]', "X", seq_ag)

        # seq_ab_v = seq_filt(seq_ab_v)
        # seq_ab_l = seq_filt(seq_ab_l)
        # seq_ag = seq_filt(seq_ag)

        # random mask off noise
        random_noise_flag = random.random()
        if random_noise_flag < 0.5:
            seq_ab_v_aug = self.seq_augmentation(seq_ab_v)
            seq_ab_l_aug = self.seq_augmentation(seq_ab_l)
        else:
            seq_ab_v_aug, seq_ab_l_aug = seq_ab_v, seq_ab_l

        ids_ag = np.zeros([max_seq_antig, 20])
        # ids_ag = np.zeros([800, 30])
        ids_ag[0:0 + len(seq_ag), :] = one_hot_encoder(s=seq_ag)

        # ids_ag[5:5 + len(onehot_encoded), :] = np.array(onehot_encoded)
        input_ids_ag = ids_ag

        # index
        # Put a space between every two letters
        ids_ab_v = tokenizer(" ".join(list(seq_ab_v)), return_tensors='pt', max_length=max_seq_antib,
                             padding='max_length',
                             truncation=True)
        ids_ab_l = tokenizer(" ".join(list(seq_ab_l)), return_tensors='pt', max_length=max_seq_antib,
                             padding='max_length',
                             truncation=True)
        ids_ab_v_aug = tokenizer(" ".join(list(seq_ab_v_aug)), return_tensors='pt', max_length=max_seq_antib,
                                 padding='max_length',
                                 truncation=True)
        ids_ab_l_aug = tokenizer(" ".join(list(seq_ab_l_aug)), return_tensors='pt', max_length=max_seq_antib,
                                 padding='max_length',
                                 truncation=True)

        # input_ids_ab_v = torch.tensor(ids_ab_v['input_ids'])
        # input_ids_ab_l = torch.tensor(ids_ab_l['input_ids'])
        input_ids_ab_v, input_ids_ab_l = ids_ab_v['input_ids'], ids_ab_l['input_ids']
        input_ids_ab_v, input_ids_ab_l = input_ids_ab_v.squeeze(0), input_ids_ab_l.squeeze(0)
        input_ids_ab_v_aug, input_ids_ab_l_aug = ids_ab_v_aug['input_ids'], ids_ab_l_aug['input_ids']
        input_ids_ab_v_aug, input_ids_ab_l_aug = input_ids_ab_v_aug.squeeze(0), input_ids_ab_l_aug.squeeze(0)

        # attention_mask_ab_v = torch.tensor(ids_ab_v['attention_mask'])
        # attention_mask_ab_l = torch.tensor(ids_ab_l['attention_mask'])
        attention_mask_ab_v, attention_mask_ab_l = ids_ab_v['attention_mask'], ids_ab_l['attention_mask']
        attention_mask_ab_v, attention_mask_ab_l = attention_mask_ab_v.squeeze(0), attention_mask_ab_l.squeeze(0)
        attention_mask_ab_v_aug, attention_mask_ab_l_aug = ids_ab_v_aug['attention_mask'], ids_ab_l_aug[
            'attention_mask']
        attention_mask_ab_v_aug, attention_mask_ab_l_aug = attention_mask_ab_v_aug.squeeze(
            0), attention_mask_ab_l_aug.squeeze(0)

        # # one-hot
        if self.onehot:
            ids_ab_v, ids_ab_l = np.zeros([max_seq_antib, 20], dtype=float), np.zeros([max_seq_antib, 20], dtype=float)
            ids_ab_v[0:0 + len(seq_ab_v), :], ids_ab_l[0:0 + len(seq_ab_l), :] = one_hot_encoder(
                s=seq_ab_v), one_hot_encoder(s=seq_ab_l)
            input_ids_ab_v, input_ids_ab_l = torch.tensor(ids_ab_v, dtype=torch.float32), torch.tensor(ids_ab_l,
                                                                                                       dtype=torch.float32)
            ids_ab_v_aug, ids_ab_l_aug = np.zeros([max_seq_antib, 20], dtype=float), np.zeros([max_seq_antib, 20],
                                                                                              dtype=float)
            ids_ab_v_aug[0:0 + len(seq_ab_v_aug), :], ids_ab_l_aug[0:0 + len(seq_ab_l_aug), :] = one_hot_encoder(
                s=seq_ab_v_aug), one_hot_encoder(s=seq_ab_l_aug)
            input_ids_ab_v_aug, input_ids_ab_l_aug = torch.tensor(ids_ab_v_aug, dtype=torch.float32), torch.tensor(
                ids_ab_l_aug, dtype=torch.float32)

        if self.rand_shift:
            shift_ab_v, shift_ab_l = np.random.randint(0, max_seq_antib - len(seq_ab_v)), np.random.randint(0,
                                                                                                            max_seq_antib - len(
                                                                                                                seq_ab_l))
            shift_ab_v_aug, shift_ab_l_aug = np.random.randint(0, max_seq_antib - len(seq_ab_v_aug)), np.random.randint(
                0, max_seq_antib - len(seq_ab_l_aug))
            shift_ag = np.random.randint(0, max_seq_antig - len(seq_ag))
        else:
            shift_ab_v, shift_ab_l = 15, 15
            shift_ab_v_aug, shift_ab_l_aug = 15, 15
            shift_ag = 15
        input_ids_ab_v = vec_shift(input_ids_ab_v, seq_shift=shift_ab_v, seq_len=len(seq_ab_v))
        attention_mask_ab_v = vec_shift(attention_mask_ab_v, seq_shift=shift_ab_v, seq_len=len(seq_ab_v))
        input_ids_ab_l = vec_shift(input_ids_ab_l, seq_shift=shift_ab_l, seq_len=len(seq_ab_l))
        attention_mask_ab_l = vec_shift(attention_mask_ab_l, seq_shift=shift_ab_l, seq_len=len(seq_ab_l))
        input_ids_ab_v_aug = vec_shift(input_ids_ab_v_aug, seq_shift=shift_ab_v_aug, seq_len=len(seq_ab_v_aug))
        attention_mask_ab_v_aug = vec_shift(attention_mask_ab_v_aug, seq_shift=shift_ab_v_aug,
                                            seq_len=len(seq_ab_v_aug))
        input_ids_ab_l_aug = vec_shift(input_ids_ab_l_aug, seq_shift=shift_ab_l_aug, seq_len=len(seq_ab_l_aug))
        attention_mask_ab_l_aug = vec_shift(attention_mask_ab_l_aug, seq_shift=shift_ab_l_aug,
                                            seq_len=len(seq_ab_l_aug))
        input_ids_ag = vec_shift(input_ids_ag, seq_shift=shift_ag, seq_len=len(seq_ag))
        # antigen_graph = 0

        # charge = data['charge']
        # hydro = data['hydro']
        # flu neu
        label = data[self.label_str]
        # try:
        #     if data[self.label_str] == 1:
        #         label = 1
        #     else:
        #         label = 0
        # except:
        #     label = -1
        #     print(label)
        # if data['value'] != 'n':
        #     value = float(data['value'])
        #     loss_ic50 = 1
        # else:
        #     value = 5000
        #     loss_ic50 = 0
        #
        # if data['Type'] == 'auto':
        #     auto = data[self.label_str]
        #     loss_auto = 1
        #     loss_main = 0
        # else:
        #     auto = 5000
        #     loss_auto = 0
        #     loss_main = 1

        loss_main = 1
        value, auto = 5000, 5000

        # print(seq_ag, seq_ab_v,seq_ab_l)
        # print(np.sum(input_ids_ag,axis=-1), np.sum(input_ids_ab_l,axis=-1))

        has_label_mask = 0 if self.sample_mode == 'no_label' else 1

        if self.onehot:
            return {'input_ids_ab_v': torch.tensor(input_ids_ab_v_aug, dtype=torch.float32),
                    'attention_mask_ab_v': torch.tensor(attention_mask_ab_v_aug, dtype=torch.float32),
                    'input_ids_ab_l': torch.tensor(input_ids_ab_l_aug, dtype=torch.float32),
                    'attention_mask_ab_l': torch.tensor(attention_mask_ab_l_aug, dtype=torch.float32),
                    'input_ids_ab_v_origin': torch.tensor(input_ids_ab_v, dtype=torch.float32),
                    'attention_mask_ab_v_origin': torch.tensor(attention_mask_ab_v, dtype=torch.float32),
                    'input_ids_ab_l_origin': torch.tensor(input_ids_ab_l, dtype=torch.float32),
                    'attention_mask_ab_l_origin': torch.tensor(attention_mask_ab_l, dtype=torch.float32),
                    'label': torch.tensor(label, dtype=torch.float32),
                    'input_ids_ag': torch.tensor(input_ids_ag, dtype=torch.float32),
                    'value': torch.tensor(value, dtype=torch.float32),
                    'auto': torch.tensor(auto, dtype=torch.float32),
                    'loss_main': torch.tensor(loss_main, dtype=torch.float32),
                    'has_label': torch.tensor(has_label_mask, dtype=torch.float32)}
            # ,torch.tensor(charge,dtype=torch.float32),torch.tensor(hydro,dtype=torch.float32)
        else:
            return {'input_ids_ab_v': input_ids_ab_v_aug,
                    'attention_mask_ab_v': torch.tensor(attention_mask_ab_v_aug),
                    'input_ids_ab_l': input_ids_ab_l_aug,
                    'attention_mask_ab_l': torch.tensor(attention_mask_ab_l_aug),
                    'input_ids_ab_v_origin': torch.tensor(input_ids_ab_v),
                    'attention_mask_ab_v_origin': torch.tensor(attention_mask_ab_v),
                    'input_ids_ab_l_origin': torch.tensor(input_ids_ab_l),
                    'attention_mask_ab_l_origin': torch.tensor(attention_mask_ab_l),
                    'label': torch.tensor(label, dtype=torch.float32),
                    'input_ids_ag': torch.tensor(input_ids_ag, dtype=torch.float32),
                    'value': torch.tensor(value, dtype=torch.float32),
                    'auto': torch.tensor(auto, dtype=torch.float32),
                    'loss_main': torch.tensor(loss_main, dtype=torch.float32),
                    'has_label': torch.tensor(has_label_mask, dtype=torch.float32)}
            # ,torch.tensor(charge,dtype=torch.float32),torch.tensor(hydro,dtype=torch.float32)


class Ab_Dataset_concat(Dataset_n):
    def __init__(self, datalist, proportions=None, sample_func=None,
                 n_samples=0, is_rand_sample=True, onehot=False, rand_shift=True):
        print('AB_Dataset initializing...')
        assert type(datalist) == list, 'Expect [datalist] to be <list> type but got <{}>.'.format(type(datalist))
        assert type(proportions) == list, 'Expect [proportions] to be <list> type but got <{}>.'.format(
            type(proportions))
        assert type(sample_func) == list, 'Expect [sample_func] to be <list> type but got <{}>.'.format(
            type(sample_func))
        assert len(datalist) == len(proportions) == len(
            sample_func), '[datalist], [proportions] and [sample_func] should have same length, however got ' \
                          'len(datalist)={}, len(proportions)={}, and len(sample_func)={}'.format(len(datalist),
                                                                                                  len(proportions),
                                                                                                  len(sample_func))
        assert n_samples > 0, 'Num of samples should be a positive number instead of {}'.format(n_samples)

        self.datalist = datalist
        self.proportions = proportions
        self.prop_norm()
        self.is_rand_sample = is_rand_sample
        self.n_samples = n_samples
        self.onehot = onehot
        self.rand_shift = rand_shift

        self.ag_head_str = "variant_seq"
        self.label_str = 'rbd'
        self.sample_func = sample_func
        self.sample_func_dict = {'sample': self.sample, 'rand_sample': self.rand_sample,
                                 'rand_sample_rand_combine': self.rand_sample_rand_combine,
                                 'rand_sample_cross_eval': self.rand_sample_cross_eval}

        # if self.mode == "train":
        #     self.sars_exp = pd.concat([self.sars_pos.copy(), self.sars_neg.copy()], ignore_index=True)

    def __len__(self):
        return self.n_samples

    def prop_norm(self):
        x = np.array(self.proportions, dtype=float)
        x /= np.sum(x)
        y = []
        for i in range(x.shape[0]):
            y.append(np.sum(x[:(i + 1)]))
        self.proportions = y
        print('proportions has been normalized to {}'.format(self.proportions))

    def sample(self, data, idx):
        return data.loc[idx].copy()

    def rand_sample(self, data, idx):
        rand_index = random.randint(0, data.shape[0] - 1)
        return data.loc[rand_index].copy()

    def rand_sample_rand_combine(self, data, idx):
        data1, data2 = data
        # this is for non-experiment data
        # fetch antigen from experiment data
        rand_index = random.randint(0, data1.shape[0] - 1)
        d = data1.loc[rand_index].copy()
        # fetch heavy/light chain from non-experiment data
        for ii in range(1000):
            rand_index = random.randint(0, data2.shape[0] - 1)
            d['Heavy'] = data2.loc[rand_index, 'Heavy']
            if type(d['Heavy']) != float:
                break
        #
        for ii in range(1000):
            rand_index = random.randint(0, data2.shape[0] - 1)
            d['Light'] = data2.loc[rand_index, 'Light']
            if type(d['Light']) != float:
                break
        #
        d[self.label_str] = 0
        return d

    def rand_sample_cross_eval(self, data, idx):
        data1, data2 = data
        n_rand = random.random()
        #
        rand_index = random.randint(0, data1.shape[0] - 1)
        d = data1.loc[rand_index].copy()
        #
        rand_index = random.randint(0, data2.shape[0] - 1)
        if n_rand < 0.5:
            d[self.ag_head_str] = data2.loc[rand_index, self.ag_head_str]
        else:
            d['Heavy'] = data2.loc[rand_index, 'Heavy']
            d['Light'] = data2.loc[rand_index, 'Light']
        d[self.label_str] = 0
        return d

    def __getitem__(self, index):
        if self.is_rand_sample:
            num_rand = random.random()
            for i, _prop in enumerate(self.proportions):
                if num_rand <= _prop:
                    data = self.sample_func_dict[self.sample_func[i]](self.datalist[i], index)
                    break
        else:
            data = self.sample_func_dict[self.sample_func[0]](self.datalist[0], index)
        print(self.datalist)
        # if self.mode == "train":
        #     num_rand = random.random()
        #     if num_rand < 0.2:
        #         data = self.rand_sample(self.sars.pos)
        #     elif num_rand < 0.8:
        #         data = self.rand_sample(self.sars.neg)
        #     else:
        #         data = self.rand_sample_rand_combine(self.sars_exp, self.sars_nonexp)
        # elif self.mode == "meta":
        #     num_rand = random.random()
        #     if num_rand < 0.25:
        #         data = self.rand_sample(self.sars_pos)
        #     else:
        #         data = self.rand_sample(self.sars_neg)
        # else:
        #     data = self.sample(self.sars_pos, index)

        # Capitalize
        seq_ab = data['Heavy'].upper() + data['Light'].upper()

        seq_ag = data[self.ag_head_str].upper()

        # Replace illegal strings
        seq_ab = re.sub(r'[UZOB*_]', "X", seq_ab)

        seq_ag = re.sub(r'[UZOB*_]', "X", seq_ag)

        # seq_ab_v = seq_filt(seq_ab_v)
        # seq_ab_l = seq_filt(seq_ab_l)
        # seq_ag = seq_filt(seq_ag)

        ids_ag = np.zeros([max_seq_antig, 20])
        # ids_ag = np.zeros([800, 30])
        ids_ag[0:0 + len(seq_ag), :] = one_hot_encoder(s=seq_ag)

        # ids_ag[5:5 + len(onehot_encoded), :] = np.array(onehot_encoded)
        input_ids_ag = ids_ag

        # index
        # Put a space between every two letters
        ids_ab = tokenizer(" ".join(list(seq_ab)), return_tensors='pt', max_length=max_seq_antib_concat,
                             padding='max_length',
                             truncation=True)


        # input_ids_ab_v = torch.tensor(ids_ab_v['input_ids'])
        # input_ids_ab_l = torch.tensor(ids_ab_l['input_ids'])
        input_ids_ab = ids_ab['input_ids']
        input_ids_ab = input_ids_ab.squeeze(0)

        # attention_mask_ab_v = torch.tensor(ids_ab_v['attention_mask'])
        # attention_mask_ab_l = torch.tensor(ids_ab_l['attention_mask'])
        attention_mask_ab = ids_ab['attention_mask']
        attention_mask_ab = attention_mask_ab.squeeze(0)

        # # one-hot
        if self.onehot:
            ids_ab_v = np.zeros([max_seq_antib_concat, 20], dtype=float)
            ids_ab_l = np.zeros([max_seq_antib_concat, 20], dtype=float)
            ids_ab[0:0 + len(seq_ab), :] = one_hot_encoder(s=seq_ab)
            input_ids_ab = ids_ab


            # print(np.shape(input_ids_ab_l))
        if self.rand_shift:
            shift_ab = np.random.randint(0, max_seq_antib_concat - len(seq_ab))
            shift_ag = np.random.randint(0, max_seq_antig - len(seq_ag))
        else:
            shift_ab = 15
            shift_ag = 15
        input_ids_ab = vec_shift(input_ids_ab, seq_shift=shift_ab, seq_len=len(seq_ab))
        attention_mask_ab = vec_shift(attention_mask_ab, seq_shift=shift_ab, seq_len=len(seq_ab))

        input_ids_ag = vec_shift(input_ids_ag, seq_shift=shift_ag, seq_len=len(seq_ag))
        # antigen_graph = 0

        # charge = data['charge']
        # hydro = data['hydro']
        # flu neu
        label=data[self.label_str]
        # try:
        #     if data[self.label_str] == 1:
        #         label = 1
        #     else:
        #         label = 0
        # except:
        #     label = -1
        #     print(label)
        # if data['value'] != 'n':
        #     value = float(data['value'])
        #     loss_ic50 = 1
        # else:
        #     value = 5000
        #     loss_ic50 = 0
        #
        # if data['Type'] == 'auto':
        #     auto = data[self.label_str]
        #     loss_auto = 1
        #     loss_main = 0
        # else:
        #     auto = 5000
        #     loss_auto = 0
        #     loss_main = 1

        loss_main = 1
        value, auto = 5000, 5000

        # print(seq_ag, seq_ab_v,seq_ab_l)
        # print(np.sum(input_ids_ag,axis=-1), np.sum(input_ids_ab_l,axis=-1))

        if self.onehot:
            return {'input_ids_ab': torch.tensor(input_ids_ab, dtype=torch.float32),
                    'attention_mask_ab': torch.tensor(attention_mask_ab, dtype=torch.float32),

                    'label': torch.tensor(label, dtype=torch.float32),
                    'input_ids_ag': torch.tensor(input_ids_ag, dtype=torch.float32),
                    'value': torch.tensor(value, dtype=torch.float32),
                    'auto': torch.tensor(auto, dtype=torch.float32),
                    'loss_main': torch.tensor(loss_main, dtype=torch.float32)}
            # ,torch.tensor(charge,dtype=torch.float32),torch.tensor(hydro,dtype=torch.float32)
        else:
            return {'input_ids_ab': torch.tensor(input_ids_ab),
                    'attention_mask_ab': torch.tensor(attention_mask_ab),

                    'label': torch.tensor(label, dtype=torch.float32),
                    'input_ids_ag': torch.tensor(input_ids_ag, dtype=torch.float32),
                    'value': torch.tensor(value, dtype=torch.float32),
                    'auto': torch.tensor(auto, dtype=torch.float32),
                    'loss_main': torch.tensor(loss_main, dtype=torch.float32)}
            # ,torch.tensor(charge,dtype=torch.float32),torch.tensor(hydro,dtype=torch.float32)


class Ab_Dataset_hiv(Dataset_n):
    def __init__(self, datalist, proportions=None, sample_func=None,
                 n_samples=0, is_rand_sample=True, onehot=False, rand_shift=True):
        print('AB_Dataset initializing...')
        assert type(datalist) == list, 'Expect [datalist] to be <list> type but got <{}>.'.format(type(datalist))
        assert type(proportions) == list, 'Expect [proportions] to be <list> type but got <{}>.'.format(
            type(proportions))
        assert type(sample_func) == list, 'Expect [sample_func] to be <list> type but got <{}>.'.format(
            type(sample_func))
        assert len(datalist) == len(proportions) == len(
            sample_func), '[datalist], [proportions] and [sample_func] should have same length, however got ' \
                          'len(datalist)={}, len(proportions)={}, and len(sample_func)={}'.format(len(datalist),
                                                                                                  len(proportions),
                                                                                                  len(sample_func))
        assert n_samples > 0, 'Num of samples should be a positive number instead of {}'.format(n_samples)

        self.datalist = datalist
        self.proportions = proportions
        self.prop_norm()
        self.is_rand_sample = is_rand_sample
        self.n_samples = n_samples
        self.onehot = onehot
        self.rand_shift = rand_shift

        self.ag_head_str = "variant_seq"
        self.label_str = 'rbd'
        self.sample_func = sample_func
        self.sample_func_dict = {'sample': self.sample, 'rand_sample': self.rand_sample,
                                 'rand_sample_rand_combine': self.rand_sample_rand_combine,
                                 'rand_sample_cross_eval': self.rand_sample_cross_eval}

        # if self.mode == "train":
        #     self.sars_exp = pd.concat([self.sars_pos.copy(), self.sars_neg.copy()], ignore_index=True)

    def __len__(self):
        return self.n_samples

    def prop_norm(self):
        x = np.array(self.proportions, dtype=float)
        x /= np.sum(x)
        y = []
        for i in range(x.shape[0]):
            y.append(np.sum(x[:(i + 1)]))
        self.proportions = y
        print('proportions has been normalized to {}'.format(self.proportions))

    def sample(self, data, idx):
        return data.loc[idx].copy()

    def rand_sample(self, data, idx):
        rand_index = random.randint(0, data.shape[0] - 1)
        return data.loc[rand_index].copy()

    def rand_sample_rand_combine(self, data, idx):
        data1, data2 = data
        # this is for non-experiment data
        # fetch antigen from experiment data
        rand_index = random.randint(0, data1.shape[0] - 1)
        d = data1.loc[rand_index].copy()
        # fetch heavy/light chain from non-experiment data
        for ii in range(1000):
            rand_index = random.randint(0, data2.shape[0] - 1)
            d['Heavy'] = data2.loc[rand_index, 'Heavy']
            if type(d['Heavy']) != float:
                break
        #
        for ii in range(1000):
            rand_index = random.randint(0, data2.shape[0] - 1)
            d['Light'] = data2.loc[rand_index, 'Light']
            if type(d['Light']) != float:
                break
        #
        d[self.label_str] = 0
        return d

    def rand_sample_cross_eval(self, data, idx):
        data1, data2 = data
        n_rand = random.random()
        #
        rand_index = random.randint(0, data1.shape[0] - 1)
        d = data1.loc[rand_index].copy()
        #
        rand_index = random.randint(0, data2.shape[0] - 1)
        if n_rand < 0.5:
            d[self.ag_head_str] = data2.loc[rand_index, self.ag_head_str]
        else:
            d['Heavy'] = data2.loc[rand_index, 'Heavy']
            d['Light'] = data2.loc[rand_index, 'Light']
        d[self.label_str] = 0
        return d

    def __getitem__(self, index):
        if self.is_rand_sample:
            num_rand = random.random()
            for i, _prop in enumerate(self.proportions):
                if num_rand <= _prop:
                    data = self.sample_func_dict[self.sample_func[i]](self.datalist[i], index)
                    break
        else:
            data = self.sample_func_dict[self.sample_func[0]](self.datalist[0], index)

        # if self.mode == "train":
        #     num_rand = random.random()
        #     if num_rand < 0.2:
        #         data = self.rand_sample(self.sars.pos)
        #     elif num_rand < 0.8:
        #         data = self.rand_sample(self.sars.neg)
        #     else:
        #         data = self.rand_sample_rand_combine(self.sars_exp, self.sars_nonexp)
        # elif self.mode == "meta":
        #     num_rand = random.random()
        #     if num_rand < 0.25:
        #         data = self.rand_sample(self.sars_pos)
        #     else:
        #         data = self.rand_sample(self.sars_neg)
        # else:
        #     data = self.sample(self.sars_pos, index)

        # Capitalize
        seq_ab_v = data['Heavy'].upper()
        seq_ab_l = data['Light'].upper()

        seq_ag = data[self.ag_head_str].upper()

        # Replace illegal strings
        seq_ab_v = re.sub(r'[UZOB*_]', "X", seq_ab_v)
        seq_ab_l = re.sub(r'[UZOB*_]', "X", seq_ab_l)

        seq_ag = re.sub(r'[UZOB*_]', "X", seq_ag)

        # seq_ab_v = seq_filt(seq_ab_v)
        # seq_ab_l = seq_filt(seq_ab_l)
        # seq_ag = seq_filt(seq_ag)

        ids_ag = np.zeros([max_seq_antig_hiv, 20])
        # ids_ag = np.zeros([800, 30])
        ids_ag[0:0 + len(seq_ag), :] = one_hot_encoder(s=seq_ag)

        # ids_ag[5:5 + len(onehot_encoded), :] = np.array(onehot_encoded)
        input_ids_ag = ids_ag

        # index
        # Put a space between every two letters
        ids_ab_v = tokenizer(" ".join(list(seq_ab_v)), return_tensors='pt', max_length=max_seq_antib,
                             padding='max_length',
                             truncation=True)
        ids_ab_l = tokenizer(" ".join(list(seq_ab_l)), return_tensors='pt', max_length=max_seq_antib,
                             padding='max_length',
                             truncation=True)

        # input_ids_ab_v = torch.tensor(ids_ab_v['input_ids'])
        # input_ids_ab_l = torch.tensor(ids_ab_l['input_ids'])
        input_ids_ab_v = ids_ab_v['input_ids']
        input_ids_ab_l = ids_ab_l['input_ids']
        input_ids_ab_v = input_ids_ab_v.squeeze(0)
        input_ids_ab_l = input_ids_ab_l.squeeze(0)

        # attention_mask_ab_v = torch.tensor(ids_ab_v['attention_mask'])
        # attention_mask_ab_l = torch.tensor(ids_ab_l['attention_mask'])
        attention_mask_ab_v = ids_ab_v['attention_mask']
        attention_mask_ab_l = ids_ab_l['attention_mask']
        attention_mask_ab_v = attention_mask_ab_v.squeeze(0)
        attention_mask_ab_l = attention_mask_ab_l.squeeze(0)

        # # one-hot
        if self.onehot:
            ids_ab_v = np.zeros([max_seq_antib, 20], dtype=float)
            ids_ab_l = np.zeros([max_seq_antib, 20], dtype=float)
            ids_ab_v[0:0 + len(seq_ab_v), :] = one_hot_encoder(s=seq_ab_v)
            ids_ab_l[0:0 + len(seq_ab_l), :] = one_hot_encoder(s=seq_ab_l)
            input_ids_ab_v = ids_ab_v
            input_ids_ab_l = ids_ab_l

            # print(np.shape(input_ids_ab_l))
        if self.rand_shift:
            shift_ab_v = np.random.randint(0, max_seq_antib - len(seq_ab_v))
            shift_ab_l = np.random.randint(0, max_seq_antib - len(seq_ab_l))
            shift_ag = np.random.randint(0, max_seq_antig_hiv - len(seq_ag))
        else:
            shift_ab_v = 15
            shift_ab_l = 15
            shift_ag = 15
        input_ids_ab_v = vec_shift(input_ids_ab_v, seq_shift=shift_ab_v, seq_len=len(seq_ab_v))
        attention_mask_ab_v = vec_shift(attention_mask_ab_v, seq_shift=shift_ab_v, seq_len=len(seq_ab_v))
        input_ids_ab_l = vec_shift(input_ids_ab_l, seq_shift=shift_ab_l, seq_len=len(seq_ab_l))
        attention_mask_ab_l = vec_shift(attention_mask_ab_l, seq_shift=shift_ab_l, seq_len=len(seq_ab_l))
        input_ids_ag = vec_shift(input_ids_ag, seq_shift=shift_ag, seq_len=len(seq_ag))
        # antigen_graph = 0

        # charge = data['charge']
        # hydro = data['hydro']
        # flu neu
        label=data[self.label_str]
        # try:
        #     if data[self.label_str] == 1:
        #         label = 1
        #     else:
        #         label = 0
        # except:
        #     label = -1
        #     print(label)
        # if data['value'] != 'n':
        #     value = float(data['value'])
        #     loss_ic50 = 1
        # else:
        #     value = 5000
        #     loss_ic50 = 0
        #
        # if data['Type'] == 'auto':
        #     auto = data[self.label_str]
        #     loss_auto = 1
        #     loss_main = 0
        # else:
        #     auto = 5000
        #     loss_auto = 0
        #     loss_main = 1

        loss_main = 1
        value, auto = 5000, 5000

        # print(seq_ag, seq_ab_v,seq_ab_l)
        # print(np.sum(input_ids_ag,axis=-1), np.sum(input_ids_ab_l,axis=-1))

        if self.onehot:
            return {'input_ids_ab_v': torch.tensor(input_ids_ab_v, dtype=torch.float32),
                    'attention_mask_ab_v': torch.tensor(attention_mask_ab_v, dtype=torch.float32),
                    'input_ids_ab_l': torch.tensor(input_ids_ab_l, dtype=torch.float32),
                    'attention_mask_ab_l': torch.tensor(attention_mask_ab_l, dtype=torch.float32),
                    'label': torch.tensor(label, dtype=torch.float32),
                    'input_ids_ag': torch.tensor(input_ids_ag, dtype=torch.float32),
                    'value': torch.tensor(value, dtype=torch.float32),
                    'auto': torch.tensor(auto, dtype=torch.float32),
                    'loss_main': torch.tensor(loss_main, dtype=torch.float32)}
            # ,torch.tensor(charge,dtype=torch.float32),torch.tensor(hydro,dtype=torch.float32)
        else:
            return {'input_ids_ab_v': torch.tensor(input_ids_ab_v),
                    'attention_mask_ab_v': torch.tensor(attention_mask_ab_v),
                    'input_ids_ab_l': torch.tensor(input_ids_ab_l),
                    'attention_mask_ab_l': torch.tensor(attention_mask_ab_l),
                    'label': torch.tensor(label, dtype=torch.float32),
                    'input_ids_ag': torch.tensor(input_ids_ag, dtype=torch.float32),
                    'value': torch.tensor(value, dtype=torch.float32),
                    'auto': torch.tensor(auto, dtype=torch.float32),
                    'loss_main': torch.tensor(loss_main, dtype=torch.float32)}
            # ,torch.tensor(charge,dtype=torch.float32),torch.tensor(hydro,dtype=torch.float32)


class Ab_Dataset_rbd(Dataset_n):
    def __init__(self, datalist, proportions=None, sample_func=None,
                 n_samples=0, is_rand_sample=True, onehot=False, rand_shift=True):
        print('AB_Dataset initializing...')
        assert type(datalist) == list, 'Expect [datalist] to be <list> type but got <{}>.'.format(type(datalist))
        assert type(proportions) == list, 'Expect [proportions] to be <list> type but got <{}>.'.format(
            type(proportions))
        assert type(sample_func) == list, 'Expect [sample_func] to be <list> type but got <{}>.'.format(
            type(sample_func))
        assert len(datalist) == len(proportions) == len(
            sample_func), '[datalist], [proportions] and [sample_func] should have same length, however got ' \
                          'len(datalist)={}, len(proportions)={}, and len(sample_func)={}'.format(len(datalist),
                                                                                                  len(proportions),
                                                                                                  len(sample_func))
        assert n_samples > 0, 'Num of samples should be a positive number instead of {}'.format(n_samples)

        self.datalist = datalist
        self.proportions = proportions
        self.prop_norm()
        self.is_rand_sample = is_rand_sample
        self.n_samples = n_samples
        self.onehot = onehot
        self.rand_shift = rand_shift

        self.ag_head_str = "variant_seq"
        self.label_str = 'rbd'
        self.sample_func = sample_func
        self.sample_func_dict = {'sample': self.sample, 'rand_sample': self.rand_sample,
                                 'rand_sample_rand_combine': self.rand_sample_rand_combine,
                                 'rand_sample_cross_eval': self.rand_sample_cross_eval}

        # if self.mode == "train":
        #     self.sars_exp = pd.concat([self.sars_pos.copy(), self.sars_neg.copy()], ignore_index=True)

    def __len__(self):
        return self.n_samples

    def prop_norm(self):
        x = np.array(self.proportions, dtype=float)
        x /= np.sum(x)
        y = []
        for i in range(x.shape[0]):
            y.append(np.sum(x[:(i + 1)]))
        self.proportions = y
        print('proportions has been normalized to {}'.format(self.proportions))

    def sample(self, data, idx):
        return data.loc[idx].copy()

    def rand_sample(self, data, idx):
        rand_index = random.randint(0, data.shape[0] - 1)
        return data.loc[rand_index].copy()

    def rand_sample_rand_combine(self, data, idx):
        data1, data2 = data
        # this is for non-experiment data
        # fetch antigen from experiment data
        rand_index = random.randint(0, data1.shape[0] - 1)
        d = data1.loc[rand_index].copy()
        # fetch heavy/light chain from non-experiment data
        for ii in range(1000):
            rand_index = random.randint(0, data2.shape[0] - 1)
            d['Heavy'] = data2.loc[rand_index, 'Heavy']
            if type(d['Heavy']) != float:
                break
        #
        for ii in range(1000):
            rand_index = random.randint(0, data2.shape[0] - 1)
            d['Light'] = data2.loc[rand_index, 'Light']
            if type(d['Light']) != float:
                break
        #
        d[self.label_str] = 0
        return d

    def rand_sample_cross_eval(self, data, idx):
        data1, data2 = data
        n_rand = random.random()
        #
        rand_index = random.randint(0, data1.shape[0] - 1)
        d = data1.loc[rand_index].copy()
        #
        rand_index = random.randint(0, data2.shape[0] - 1)
        if n_rand < 0.5:
            d[self.ag_head_str] = data2.loc[rand_index, self.ag_head_str]
        else:
            d['Heavy'] = data2.loc[rand_index, 'Heavy']
            d['Light'] = data2.loc[rand_index, 'Light']
        d[self.label_str] = 0
        return d

    def __getitem__(self, index):
        if self.is_rand_sample:
            num_rand = random.random()
            for i, _prop in enumerate(self.proportions):
                if num_rand <= _prop:
                    data = self.sample_func_dict[self.sample_func[i]](self.datalist[i], index)
                    break
        else:
            data = self.sample_func_dict[self.sample_func[0]](self.datalist[0], index)

        # if self.mode == "train":
        #     num_rand = random.random()
        #     if num_rand < 0.2:
        #         data = self.rand_sample(self.sars.pos)
        #     elif num_rand < 0.8:
        #         data = self.rand_sample(self.sars.neg)
        #     else:
        #         data = self.rand_sample_rand_combine(self.sars_exp, self.sars_nonexp)
        # elif self.mode == "meta":
        #     num_rand = random.random()
        #     if num_rand < 0.25:
        #         data = self.rand_sample(self.sars_pos)
        #     else:
        #         data = self.rand_sample(self.sars_neg)
        # else:
        #     data = self.sample(self.sars_pos, index)

        # Capitalize
        seq_ab_v = data['Heavy'].upper()
        seq_ab_l = data['Light'].upper()

        seq_ag = data[self.ag_head_str].upper()

        # Replace illegal strings
        seq_ab_v = re.sub(r'[UZOB*_]', "X", seq_ab_v)
        seq_ab_l = re.sub(r'[UZOB*_]', "X", seq_ab_l)

        seq_ag = re.sub(r'[UZOB*_]', "X", seq_ag)

        # seq_ab_v = seq_filt(seq_ab_v)
        # seq_ab_l = seq_filt(seq_ab_l)
        # seq_ag = seq_filt(seq_ag)

        ids_ag = np.zeros([max_seq_antig_rbd, 20])
        # ids_ag = np.zeros([800, 30])
        ids_ag[0:0 + len(seq_ag), :] = one_hot_encoder(s=seq_ag)

        # ids_ag[5:5 + len(onehot_encoded), :] = np.array(onehot_encoded)
        input_ids_ag = ids_ag

        # index
        # Put a space between every two letters
        ids_ab_v = tokenizer(" ".join(list(seq_ab_v)), return_tensors='pt', max_length=max_seq_antib,
                             padding='max_length',
                             truncation=True)
        ids_ab_l = tokenizer(" ".join(list(seq_ab_l)), return_tensors='pt', max_length=max_seq_antib,
                             padding='max_length',
                             truncation=True)

        # input_ids_ab_v = torch.tensor(ids_ab_v['input_ids'])
        # input_ids_ab_l = torch.tensor(ids_ab_l['input_ids'])
        input_ids_ab_v = ids_ab_v['input_ids']
        input_ids_ab_l = ids_ab_l['input_ids']
        input_ids_ab_v = input_ids_ab_v.squeeze(0)
        input_ids_ab_l = input_ids_ab_l.squeeze(0)

        # attention_mask_ab_v = torch.tensor(ids_ab_v['attention_mask'])
        # attention_mask_ab_l = torch.tensor(ids_ab_l['attention_mask'])
        attention_mask_ab_v = ids_ab_v['attention_mask']
        attention_mask_ab_l = ids_ab_l['attention_mask']
        attention_mask_ab_v = attention_mask_ab_v.squeeze(0)
        attention_mask_ab_l = attention_mask_ab_l.squeeze(0)

        # # one-hot
        if self.onehot:
            ids_ab_v = np.zeros([max_seq_antib, 20], dtype=float)
            ids_ab_l = np.zeros([max_seq_antib, 20], dtype=float)
            ids_ab_v[0:0 + len(seq_ab_v), :] = one_hot_encoder(s=seq_ab_v)
            ids_ab_l[0:0 + len(seq_ab_l), :] = one_hot_encoder(s=seq_ab_l)
            input_ids_ab_v = ids_ab_v
            input_ids_ab_l = ids_ab_l

            # print(np.shape(input_ids_ab_l))
        if self.rand_shift:
            shift_ab_v = np.random.randint(0, max_seq_antib - len(seq_ab_v))
            shift_ab_l = np.random.randint(0, max_seq_antib - len(seq_ab_l))
            shift_ag = np.random.randint(0, max_seq_antig_rbd - len(seq_ag))
        else:
            shift_ab_v = 15
            shift_ab_l = 15
            shift_ag = 15
        input_ids_ab_v = vec_shift(input_ids_ab_v, seq_shift=shift_ab_v, seq_len=len(seq_ab_v))
        attention_mask_ab_v = vec_shift(attention_mask_ab_v, seq_shift=shift_ab_v, seq_len=len(seq_ab_v))
        input_ids_ab_l = vec_shift(input_ids_ab_l, seq_shift=shift_ab_l, seq_len=len(seq_ab_l))
        attention_mask_ab_l = vec_shift(attention_mask_ab_l, seq_shift=shift_ab_l, seq_len=len(seq_ab_l))
        input_ids_ag = vec_shift(input_ids_ag, seq_shift=shift_ag, seq_len=len(seq_ag))
        # antigen_graph = 0

        # charge = data['charge']
        # hydro = data['hydro']
        # flu neu
        label=data[self.label_str]
        # try:
        #     if data[self.label_str] == 1:
        #         label = 1
        #     else:
        #         label = 0
        # except:
        #     label = -1
        #     print(label)
        # if data['value'] != 'n':
        #     value = float(data['value'])
        #     loss_ic50 = 1
        # else:
        #     value = 5000
        #     loss_ic50 = 0
        #
        # if data['Type'] == 'auto':
        #     auto = data[self.label_str]
        #     loss_auto = 1
        #     loss_main = 0
        # else:
        #     auto = 5000
        #     loss_auto = 0
        #     loss_main = 1

        loss_main = 1
        value, auto = 5000, 5000

        # print(seq_ag, seq_ab_v,seq_ab_l)
        # print(np.sum(input_ids_ag,axis=-1), np.sum(input_ids_ab_l,axis=-1))

        if self.onehot:
            return {'input_ids_ab_v': torch.tensor(input_ids_ab_v, dtype=torch.float32),
                    'attention_mask_ab_v': torch.tensor(attention_mask_ab_v, dtype=torch.float32),
                    'input_ids_ab_l': torch.tensor(input_ids_ab_l, dtype=torch.float32),
                    'attention_mask_ab_l': torch.tensor(attention_mask_ab_l, dtype=torch.float32),
                    'label': torch.tensor(label, dtype=torch.float32),
                    'input_ids_ag': torch.tensor(input_ids_ag, dtype=torch.float32),
                    'value': torch.tensor(value, dtype=torch.float32),
                    'auto': torch.tensor(auto, dtype=torch.float32),
                    'loss_main': torch.tensor(loss_main, dtype=torch.float32)}
            # ,torch.tensor(charge,dtype=torch.float32),torch.tensor(hydro,dtype=torch.float32)
        else:
            return {'input_ids_ab_v': torch.tensor(input_ids_ab_v),
                    'attention_mask_ab_v': torch.tensor(attention_mask_ab_v),
                    'input_ids_ab_l': torch.tensor(input_ids_ab_l),
                    'attention_mask_ab_l': torch.tensor(attention_mask_ab_l),
                    'label': torch.tensor(label, dtype=torch.float32),
                    'input_ids_ag': torch.tensor(input_ids_ag, dtype=torch.float32),
                    'value': torch.tensor(value, dtype=torch.float32),
                    'auto': torch.tensor(auto, dtype=torch.float32),
                    'loss_main': torch.tensor(loss_main, dtype=torch.float32)}
            # ,torch.tensor(charge,dtype=torch.float32),torch.tensor(hydro,dtype=torch.float32)


class Ab_Dataset_flu(Dataset_n):
    def __init__(self, datalist, proportions=None, sample_func=None,
                 n_samples=0, is_rand_sample=True, onehot=False, rand_shift=True):
        print('AB_Dataset initializing...')
        assert type(datalist) == list, 'Expect [datalist] to be <list> type but got <{}>.'.format(type(datalist))
        assert type(proportions) == list, 'Expect [proportions] to be <list> type but got <{}>.'.format(
            type(proportions))
        assert type(sample_func) == list, 'Expect [sample_func] to be <list> type but got <{}>.'.format(
            type(sample_func))
        assert len(datalist) == len(proportions) == len(
            sample_func), '[datalist], [proportions] and [sample_func] should have same length, however got ' \
                          'len(datalist)={}, len(proportions)={}, and len(sample_func)={}'.format(len(datalist),
                                                                                                  len(proportions),
                                                                                                  len(sample_func))
        assert n_samples > 0, 'Num of samples should be a positive number instead of {}'.format(n_samples)

        self.datalist = datalist
        self.proportions = proportions
        self.prop_norm()
        self.is_rand_sample = is_rand_sample
        self.n_samples = n_samples
        self.onehot = onehot
        self.rand_shift = rand_shift

        self.ag_head_str = "variant_seq"
        self.label_str = 'rbd'
        self.sample_func = sample_func
        self.sample_func_dict = {'sample': self.sample, 'rand_sample': self.rand_sample,
                                 'rand_sample_rand_combine': self.rand_sample_rand_combine,
                                 'rand_sample_cross_eval': self.rand_sample_cross_eval}

        # if self.mode == "train":
        #     self.sars_exp = pd.concat([self.sars_pos.copy(), self.sars_neg.copy()], ignore_index=True)

    def __len__(self):
        return self.n_samples

    def prop_norm(self):
        x = np.array(self.proportions, dtype=float)
        x /= np.sum(x)
        y = []
        for i in range(x.shape[0]):
            y.append(np.sum(x[:(i + 1)]))
        self.proportions = y
        print('proportions has been normalized to {}'.format(self.proportions))

    def sample(self, data, idx):
        return data.loc[idx].copy()

    def rand_sample(self, data, idx):
        rand_index = random.randint(0, data.shape[0] - 1)
        return data.loc[rand_index].copy()

    def rand_sample_rand_combine(self, data, idx):
        data1, data2 = data
        # this is for non-experiment data
        # fetch antigen from experiment data
        rand_index = random.randint(0, data1.shape[0] - 1)
        d = data1.loc[rand_index].copy()
        # fetch heavy/light chain from non-experiment data
        for ii in range(1000):
            rand_index = random.randint(0, data2.shape[0] - 1)
            d['Heavy'] = data2.loc[rand_index, 'Heavy']
            if type(d['Heavy']) != float:
                break
        #
        for ii in range(1000):
            rand_index = random.randint(0, data2.shape[0] - 1)
            d['Light'] = data2.loc[rand_index, 'Light']
            if type(d['Light']) != float:
                break
        #
        d[self.label_str] = 0
        return d

    def rand_sample_cross_eval(self, data, idx):
        data1, data2 = data
        n_rand = random.random()
        #
        rand_index = random.randint(0, data1.shape[0] - 1)
        d = data1.loc[rand_index].copy()
        #
        rand_index = random.randint(0, data2.shape[0] - 1)
        if n_rand < 0.5:
            d[self.ag_head_str] = data2.loc[rand_index, self.ag_head_str]
        else:
            d['Heavy'] = data2.loc[rand_index, 'Heavy']
            d['Light'] = data2.loc[rand_index, 'Light']
        d[self.label_str] = 0
        return d

    def __getitem__(self, index):
        if self.is_rand_sample:
            num_rand = random.random()
            for i, _prop in enumerate(self.proportions):
                if num_rand <= _prop:
                    data = self.sample_func_dict[self.sample_func[i]](self.datalist[i], index)
                    break
        else:
            data = self.sample_func_dict[self.sample_func[0]](self.datalist[0], index)

        # if self.mode == "train":
        #     num_rand = random.random()
        #     if num_rand < 0.2:
        #         data = self.rand_sample(self.sars.pos)
        #     elif num_rand < 0.8:
        #         data = self.rand_sample(self.sars.neg)
        #     else:
        #         data = self.rand_sample_rand_combine(self.sars_exp, self.sars_nonexp)
        # elif self.mode == "meta":
        #     num_rand = random.random()
        #     if num_rand < 0.25:
        #         data = self.rand_sample(self.sars_pos)
        #     else:
        #         data = self.rand_sample(self.sars_neg)
        # else:
        #     data = self.sample(self.sars_pos, index)

        # Capitalize
        seq_ab_v = data['Heavy'].upper()
        seq_ab_l = data['Light'].upper()

        seq_ag = data[self.ag_head_str].upper()

        # Replace illegal strings
        seq_ab_v = re.sub(r'[UZOB*_]', "X", seq_ab_v)
        seq_ab_l = re.sub(r'[UZOB*_]', "X", seq_ab_l)

        seq_ag = re.sub(r'[UZOB*_]', "X", seq_ag)

        # seq_ab_v = seq_filt(seq_ab_v)
        # seq_ab_l = seq_filt(seq_ab_l)
        # seq_ag = seq_filt(seq_ag)

        ids_ag = np.zeros([max_seq_antig_flu, 20])
        # ids_ag = np.zeros([800, 30])
        ids_ag[0:0 + len(seq_ag), :] = one_hot_encoder(s=seq_ag)

        # ids_ag[5:5 + len(onehot_encoded), :] = np.array(onehot_encoded)
        input_ids_ag = ids_ag

        # index
        # Put a space between every two letters
        ids_ab_v = tokenizer(" ".join(list(seq_ab_v)), return_tensors='pt', max_length=max_seq_antib,
                             padding='max_length',
                             truncation=True)
        ids_ab_l = tokenizer(" ".join(list(seq_ab_l)), return_tensors='pt', max_length=max_seq_antib,
                             padding='max_length',
                             truncation=True)

        # input_ids_ab_v = torch.tensor(ids_ab_v['input_ids'])
        # input_ids_ab_l = torch.tensor(ids_ab_l['input_ids'])
        input_ids_ab_v = ids_ab_v['input_ids']
        input_ids_ab_l = ids_ab_l['input_ids']
        input_ids_ab_v = input_ids_ab_v.squeeze(0)
        input_ids_ab_l = input_ids_ab_l.squeeze(0)

        # attention_mask_ab_v = torch.tensor(ids_ab_v['attention_mask'])
        # attention_mask_ab_l = torch.tensor(ids_ab_l['attention_mask'])
        attention_mask_ab_v = ids_ab_v['attention_mask']
        attention_mask_ab_l = ids_ab_l['attention_mask']
        attention_mask_ab_v = attention_mask_ab_v.squeeze(0)
        attention_mask_ab_l = attention_mask_ab_l.squeeze(0)

        # # one-hot
        if self.onehot:
            ids_ab_v = np.zeros([max_seq_antib, 20], dtype=float)
            ids_ab_l = np.zeros([max_seq_antib, 20], dtype=float)
            ids_ab_v[0:0 + len(seq_ab_v), :] = one_hot_encoder(s=seq_ab_v)
            ids_ab_l[0:0 + len(seq_ab_l), :] = one_hot_encoder(s=seq_ab_l)
            input_ids_ab_v = ids_ab_v
            input_ids_ab_l = ids_ab_l

            # print(np.shape(input_ids_ab_l))
        if self.rand_shift:
            shift_ab_v = np.random.randint(0, max_seq_antib - len(seq_ab_v))
            shift_ab_l = np.random.randint(0, max_seq_antib - len(seq_ab_l))
            shift_ag = np.random.randint(0, max_seq_antig_flu - len(seq_ag))
        else:
            shift_ab_v = 15
            shift_ab_l = 15
            shift_ag = 15
        input_ids_ab_v = vec_shift(input_ids_ab_v, seq_shift=shift_ab_v, seq_len=len(seq_ab_v))
        attention_mask_ab_v = vec_shift(attention_mask_ab_v, seq_shift=shift_ab_v, seq_len=len(seq_ab_v))
        input_ids_ab_l = vec_shift(input_ids_ab_l, seq_shift=shift_ab_l, seq_len=len(seq_ab_l))
        attention_mask_ab_l = vec_shift(attention_mask_ab_l, seq_shift=shift_ab_l, seq_len=len(seq_ab_l))
        input_ids_ag = vec_shift(input_ids_ag, seq_shift=shift_ag, seq_len=len(seq_ag))
        # antigen_graph = 0

        # charge = data['charge']
        # hydro = data['hydro']
        # flu neu
        label=data[self.label_str]
        # try:
        #     if data[self.label_str] == 1:
        #         label = 1
        #     else:
        #         label = 0
        # except:
        #     label = -1
        #     print(label)
        # if data['value'] != 'n':
        #     value = float(data['value'])
        #     loss_ic50 = 1
        # else:
        #     value = 5000
        #     loss_ic50 = 0
        #
        # if data['Type'] == 'auto':
        #     auto = data[self.label_str]
        #     loss_auto = 1
        #     loss_main = 0
        # else:
        #     auto = 5000
        #     loss_auto = 0
        #     loss_main = 1

        loss_main = 1
        value, auto = 5000, 5000

        # print(seq_ag, seq_ab_v,seq_ab_l)
        # print(np.sum(input_ids_ag,axis=-1), np.sum(input_ids_ab_l,axis=-1))

        if self.onehot:
            return {'input_ids_ab_v': torch.tensor(input_ids_ab_v, dtype=torch.float32),
                    'attention_mask_ab_v': torch.tensor(attention_mask_ab_v, dtype=torch.float32),
                    'input_ids_ab_l': torch.tensor(input_ids_ab_l, dtype=torch.float32),
                    'attention_mask_ab_l': torch.tensor(attention_mask_ab_l, dtype=torch.float32),
                    'label': torch.tensor(label, dtype=torch.float32),
                    'input_ids_ag': torch.tensor(input_ids_ag, dtype=torch.float32),
                    'value': torch.tensor(value, dtype=torch.float32),
                    'auto': torch.tensor(auto, dtype=torch.float32),
                    'loss_main': torch.tensor(loss_main, dtype=torch.float32)}
            # ,torch.tensor(charge,dtype=torch.float32),torch.tensor(hydro,dtype=torch.float32)
        else:
            return {'input_ids_ab_v': torch.tensor(input_ids_ab_v),
                    'attention_mask_ab_v': torch.tensor(attention_mask_ab_v),
                    'input_ids_ab_l': torch.tensor(input_ids_ab_l),
                    'attention_mask_ab_l': torch.tensor(attention_mask_ab_l),
                    'label': torch.tensor(label, dtype=torch.float32),
                    'input_ids_ag': torch.tensor(input_ids_ag, dtype=torch.float32),
                    'value': torch.tensor(value, dtype=torch.float32),
                    'auto': torch.tensor(auto, dtype=torch.float32),
                    'loss_main': torch.tensor(loss_main, dtype=torch.float32)}
            # ,torch.tensor(charge,dtype=torch.float32),torch.tensor(hydro,dtype=torch.float32)


class Ab_Dataset_mean_pooling(Dataset_n):
    def __init__(self, datalist, proportions=None, sample_func=None,
                 n_samples=0, is_rand_sample=True, onehot=False, rand_shift=True):
        print('AB_Dataset initializing...')
        assert type(datalist) == list, 'Expect [datalist] to be <list> type but got <{}>.'.format(type(datalist))
        assert type(proportions) == list, 'Expect [proportions] to be <list> type but got <{}>.'.format(
            type(proportions))
        assert type(sample_func) == list, 'Expect [sample_func] to be <list> type but got <{}>.'.format(
            type(sample_func))
        assert len(datalist) == len(proportions) == len(
            sample_func), '[datalist], [proportions] and [sample_func] should have same length, however got ' \
                          'len(datalist)={}, len(proportions)={}, and len(sample_func)={}'.format(len(datalist),
                                                                                                  len(proportions),
                                                                                                  len(sample_func))
        assert n_samples > 0, 'Num of samples should be a positive number instead of {}'.format(n_samples)

        self.datalist = datalist
        self.proportions = proportions
        self.prop_norm()
        self.is_rand_sample = is_rand_sample
        self.n_samples = n_samples
        self.onehot = onehot
        self.rand_shift = rand_shift

        self.ag_head_str = "variant_seq"
        self.label_str = 'rbd'
        self.sample_func = sample_func
        self.sample_func_dict = {'sample': self.sample, 'rand_sample': self.rand_sample,
                                 'rand_sample_rand_combine': self.rand_sample_rand_combine,
                                 'rand_sample_cross_eval': self.rand_sample_cross_eval}

        # if self.mode == "train":
        #     self.sars_exp = pd.concat([self.sars_pos.copy(), self.sars_neg.copy()], ignore_index=True)

    def __len__(self):
        return self.n_samples

    def prop_norm(self):
        x = np.array(self.proportions, dtype=float)
        x /= np.sum(x)
        y = []
        for i in range(x.shape[0]):
            y.append(np.sum(x[:(i + 1)]))
        self.proportions = y
        print('proportions has been normalized to {}'.format(self.proportions))

    def sample(self, data, idx):
        return data.loc[idx].copy()

    def rand_sample(self, data, idx):
        rand_index = random.randint(0, data.shape[0] - 1)
        return data.loc[rand_index].copy()

    def rand_sample_rand_combine(self, data, idx):
        data1, data2 = data
        # this is for non-experiment data
        # fetch antigen from experiment data
        rand_index = random.randint(0, data1.shape[0] - 1)
        d = data1.loc[rand_index].copy()
        # fetch heavy/light chain from non-experiment data
        for ii in range(1000):
            rand_index = random.randint(0, data2.shape[0] - 1)
            d['Heavy'] = data2.loc[rand_index, 'Heavy']
            if type(d['Heavy']) != float:
                break
        #
        for ii in range(1000):
            rand_index = random.randint(0, data2.shape[0] - 1)
            d['Light'] = data2.loc[rand_index, 'Light']
            if type(d['Light']) != float:
                break
        #
        d[self.label_str] = 0
        return d

    def rand_sample_cross_eval(self, data, idx):
        data1, data2 = data
        n_rand = random.random()
        #
        rand_index = random.randint(0, data1.shape[0] - 1)
        d = data1.loc[rand_index].copy()
        #
        rand_index = random.randint(0, data2.shape[0] - 1)
        if n_rand < 0.5:
            d[self.ag_head_str] = data2.loc[rand_index, self.ag_head_str]
        else:
            d['Heavy'] = data2.loc[rand_index, 'Heavy']
            d['Light'] = data2.loc[rand_index, 'Light']
        d[self.label_str] = 0
        return d

    def __getitem__(self, index):
        if self.is_rand_sample:
            num_rand = random.random()
            for i, _prop in enumerate(self.proportions):
                if num_rand <= _prop:
                    data = self.sample_func_dict[self.sample_func[i]](self.datalist[i], index)
                    break
        else:
            data = self.sample_func_dict[self.sample_func[0]](self.datalist[0], index)

        # if self.mode == "train":
        #     num_rand = random.random()
        #     if num_rand < 0.2:
        #         data = self.rand_sample(self.sars.pos)
        #     elif num_rand < 0.8:
        #         data = self.rand_sample(self.sars.neg)
        #     else:
        #         data = self.rand_sample_rand_combine(self.sars_exp, self.sars_nonexp)
        # elif self.mode == "meta":
        #     num_rand = random.random()
        #     if num_rand < 0.25:
        #         data = self.rand_sample(self.sars_pos)
        #     else:
        #         data = self.rand_sample(self.sars_neg)
        # else:
        #     data = self.sample(self.sars_pos, index)

        # Capitalize
        seq_ab_v = data['Heavy'].upper()
        seq_ab_l = data['Light'].upper()

        seq_ag = data[self.ag_head_str].upper()

        # Replace illegal strings
        seq_ab_v = re.sub(r'[UZOB*_]', "X", seq_ab_v)
        seq_ab_l = re.sub(r'[UZOB*_]', "X", seq_ab_l)

        seq_ag = re.sub(r'[UZOB*_]', "X", seq_ag)

        # index
        # Put a space between every two letters
        ids_ab_v = tokenizer(" ".join(list(seq_ab_v)), return_tensors='pt', max_length=max_seq_antib,
                             padding='max_length',
                             truncation=True)
        ids_ab_l = tokenizer(" ".join(list(seq_ab_l)), return_tensors='pt', max_length=max_seq_antib,
                             padding='max_length',
                             truncation=True)
        ids_ag = tokenizer(" ".join(list(seq_ag)), return_tensors='pt', max_length=max_seq_antig,
                             padding='max_length',
                             truncation=True)


        input_ids_ab_v = ids_ab_v['input_ids']
        input_ids_ab_v = input_ids_ab_v.squeeze(0)
        input_ids_ab_l = ids_ab_l['input_ids']
        input_ids_ab_l = input_ids_ab_l.squeeze(0)
        input_ids_ag = ids_ag['input_ids']
        input_ids_ag = input_ids_ag.squeeze(0)

        attention_mask_ab_v = ids_ab_v['attention_mask']
        attention_mask_ab_v = attention_mask_ab_v.squeeze(0)
        attention_mask_ab_l = ids_ab_l['attention_mask']
        attention_mask_ab_l = attention_mask_ab_l.squeeze(0)
        attention_mask_ag = ids_ag['attention_mask']
        attention_mask_ag = attention_mask_ag.squeeze(0)

        # # one-hot
        label=data[self.label_str]

        return {'input_ids_ab_v': torch.tensor(input_ids_ab_v),
                'attention_mask_ab_v': torch.tensor(attention_mask_ab_v),
                'input_ids_ab_l': torch.tensor(input_ids_ab_l),
                'attention_mask_ab_l': torch.tensor(attention_mask_ab_l),
                'label': torch.tensor(label, dtype=torch.float32),
                'input_ids_ag': torch.tensor(input_ids_ag),
                'attention_mask_ag': torch.tensor(attention_mask_ag)}

