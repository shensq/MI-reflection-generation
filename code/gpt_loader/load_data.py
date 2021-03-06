import torch
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable
import json
import random
import sys
import pickle
from tqdm import tqdm
from collections import deque
import copy
sys.path.append("..")
from utils import text_standardize


USE_CUDA = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor


# ==== Code for data loading =====
class GptDataset(Dataset):
    """Take a list of samples with form [[x,...],y,meta]
    """
    # need 3 special tokens
    # # as <ref start> 2
    # $ as <speaker1> 3
    # % as <speaker2> 4
    # '<|endoftext|>' as <eos> 50256
    def _split(self,x_y_meta):
        x_all = []
        y_all = []
        meta_all = []
        for x,y,meta in x_y_meta:
            meta_all.append(meta)
            x_all.append([self.tokenizer.encode(text_standardize(x_i)) for x_i in x])
            y_all.append(self.tokenizer.encode(text_standardize(y)))

        return x_all,y_all,meta_all
    
    def _filter(self,x_all,y_all,meta_all,filter_mode=None):
        allowed_pattern = ['SR_only','CR_only','Smoking_only','Diet_only']
        data = zip(x_all,y_all,meta_all)
        if filter_mode not in allowed_pattern:
            data_filt = data
        if filter_mode=='SR_only':
            data_filt = [x for x in data if x[2][2]=='SR']
        if filter_mode=='CR_only':
            data_filt = [x for x in data if x[2][2]=='CR']
        if filter_mode=='Smoking_only':
            data_filt = [x for x in data if x[2][1]=='Smoking cessation']
        if filter_mode=='Diet_only':
            data_filt = [x for x in data if x[2][1]=='Weight management']
        x_filt,y_filt,meta_filt = zip(*data_filt)
        return x_filt, y_filt, meta_filt

    def __init__(self,x_y_meta,tokenizer,filter_mode=None,num_turns=5):
        
        self.x_y_meta = x_y_meta
        self.num_turns = num_turns
        self.tokenizer = tokenizer
        self.x_encoded,self.y_encoded,self.meta = self._split(x_y_meta)
        self.x_encoded,self.y_encoded,self.meta = self._filter(self.x_encoded,self.y_encoded,self.meta,filter_mode)
        self.ref_start, self.speaker1,self.speaker2,self.eos = 2,3,4,50256

    def __getitem__(self,index):
        x = []
        type_x = []
        lm_x = []
        is_speaker1 = bool(self.num_turns % 2) # which speaker start the conversation

        for utt in self.x_encoded[index][-self.num_turns:]:
            if is_speaker1: # add the prefix special token for each utterance
                x+=[self.speaker1]
                type_x += [self.speaker1]*(len(utt)+1)
            else:
                x+=[self.speaker2]
                type_x += [self.speaker2]*(len(utt)+1)
            x += utt
            is_speaker1 = not is_speaker1
        lm_x += [-1]*len(x) # all position for the input is masked for loss calculation

        total_input_length = len(x)

        x += [self.ref_start] + self.y_encoded[index] + [self.eos]

        type_x += [self.ref_start]*(len(self.y_encoded[index])+2)
        lm_x += [-1] + self.y_encoded[index] + [self.eos]
        position_x = list(range(len(x)))

        x = torch.Tensor(x)
        type_x = torch.Tensor(type_x)
        position_x = torch.Tensor(position_x)
        lm_x = torch.Tensor(lm_x)
        x_len = x.shape[0]
        
        return x,type_x,position_x,lm_x,total_input_length,self.meta[index]

    def __len__(self):
        return len(self.x_encoded)

class GptDataset_aug(Dataset):
    def _split(self,x_y_meta):
        x_all = []
        y_all = []
        meta_all = []
        aug_all = []
        for x,y,meta,aug in x_y_meta: 
            meta_all.append(meta)
            x_all.append([self.tokenizer.encode(text_standardize(x_i)) for x_i in x])
            y_all.append(self.tokenizer.encode(text_standardize(y)))
            aug_all.append(self.tokenizer.encode(text_standardize(aug)))
        return x_all,y_all,meta_all,aug_all

    def __init__(self,x_y_meta,tokenizer,num_turns=5):
        self.x_y_meta = x_y_meta
        self.num_turns = num_turns
        self.tokenizer = tokenizer
        self.x_encoded,self.y_encoded,self.meta,self.aug_encoded = self._split(x_y_meta)
        self.ref_start, self.speaker1,self.speaker2,self.eos = 2,3,4,50256
        self.augment = 5

    def __getitem__(self,index):
        x = []
        type_x = []
        lm_x = []

        x += [self.augment] + self.aug_encoded[index]
        type_x += [self.augment] * len(x)

        is_speaker1 = bool(self.num_turns % 2) # which speaker start the conversation

        for utt in self.x_encoded[index][-self.num_turns:]:
            if is_speaker1: # add the prefix special token for each utterance
                x+=[self.speaker1]
                type_x += [self.speaker1]*(len(utt)+1)
            else:
                x+=[self.speaker2]
                type_x += [self.speaker2]*(len(utt)+1)
            x += utt
            is_speaker1 = not is_speaker1
        lm_x += [-1]*len(x) # all position for the input is masked for loss calculation

        total_input_length = len(x)

        x += [self.ref_start] + self.y_encoded[index] + [self.eos]

        type_x += [self.ref_start]*(len(self.y_encoded[index])+2)
        lm_x += [-1] + self.y_encoded[index] + [self.eos]
        position_x = list(range(len(x)))

        x = torch.Tensor(x)
        type_x = torch.Tensor(type_x)
        position_x = torch.Tensor(position_x)
        lm_x = torch.Tensor(lm_x)
        x_len = x.shape[0]
        
        return x,type_x,position_x,lm_x,total_input_length,self.meta[index]
    def __len__(self):
        return len(self.x_encoded)

def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (src_seq, trg_seq).
    We should build a custom collate_fn rather than using default collate_fn,
    because merging sequences (including padding) is not supported in default.
    Seqeuences are padded to the maximum length of mini-batch sequences (dynamic padding).
    Args:
        data: list of tuple (src_seq, trg_seq).
            - src_seq: torch tensor of shape (?); variable length.
            - trg_seq: torch tensor of shape (?); variable length.
    Returns:
        src_seqs: torch tensor of shape (batch_size, padded_length).
        src_lengths: list of length (batch_size); valid length for each padded source sequence.
        trg_seqs: torch tensor of shape (batch_size, padded_length).
        trg_lengths: list of length (batch_size); valid length for each padded target sequence.
    """
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths
    def merge_matrix(matrices):
        max_size = max([m.shape[-1] for m in attention_mask])
        padded_matrices = torch.zeros(len(matrices), 1,  max_size, max_size)
        for i,m in enumerate(matrices):
            m_size = m.shape[-1]
            padded_matrices[i,:,:m_size,:m_size] = m
        return padded_matrices


    # sort a list by sequence length (descending order) to use pack_padded_sequence
    data.sort(key=lambda x: len(x[0]), reverse=True)

    # seperate source and target sequences
    src_seqs, trg_seqs, pos_seqs,lm_seqs,total_input_length,attention_mask = zip(*data)

    # merge sequences (from tuple of 1D tensor to 2D tensor)
    src_seqs, src_lengths = merge(src_seqs)
    trg_seqs, trg_lengths = merge(trg_seqs)
    pos_seqs, pos_lengths = merge(pos_seqs)
    lm_seqs, lm_lengths = merge(lm_seqs)
    if type(attention_mask[0]) is not list:
        attention_mask = merge_matrix(attention_mask)
    if USE_CUDA:
        src_seqs = src_seqs.cuda()
        trg_seqs = trg_seqs.cuda()
        pos_seqs = pos_seqs.cuda()
        lm_seqs = lm_seqs.cuda()
        attention_mask = attention_mask.cuda()
    return Variable(LongTensor(src_seqs)), Variable(LongTensor(trg_seqs)), Variable(LongTensor(pos_seqs)),Variable(LongTensor(lm_seqs)), total_input_length, attention_mask

def collate_fn_conditional(data):
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    # sort a list by sequence length (descending order) to use pack_padded_sequence
    data.sort(key=lambda x: len(x[0]), reverse=True)

    # seperate source and target sequences
    src_seqs, trg_seqs, pos_seqs,lm_seqs,total_input_length, meta = zip(*data)

    # merge sequences (from tuple of 1D tensor to 2D tensor)
    src_seqs, src_lengths = merge(src_seqs)
    trg_seqs, trg_lengths = merge(trg_seqs)
    pos_seqs, pos_lengths = merge(pos_seqs)
    lm_seqs, lm_lengths = merge(lm_seqs)

    if USE_CUDA:
        src_seqs = src_seqs.cuda()
        trg_seqs = trg_seqs.cuda()
        pos_seqs = pos_seqs.cuda()
        lm_seqs = lm_seqs.cuda()
    return Variable(LongTensor(src_seqs)), Variable(LongTensor(trg_seqs)), Variable(LongTensor(pos_seqs)),Variable(LongTensor(lm_seqs)), total_input_length, meta

def collate_fn_nli(data):
    """Creates mini-batch tensors from the list of tuples (src_seq, trg_seq).
    We should build a custom collate_fn rather than using default collate_fn,
    because merging sequences (including padding) is not supported in default.
    Seqeuences are padded to the maximum length of mini-batch sequences (dynamic padding).
    Args:
        data: list of tuple (src_seq, trg_seq).
            - src_seq: torch tensor of shape (?); variable length.
            - trg_seq: torch tensor of shape (?); variable length.
    Returns:
        src_seqs: torch tensor of shape (batch_size, padded_length).
        src_lengths: list of length (batch_size); valid length for each padded source sequence.
        trg_seqs: torch tensor of shape (batch_size, padded_length).
        trg_lengths: list of length (batch_size); valid length for each padded target sequence.
    """
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    # sort a list by sequence length (descending order) to use pack_padded_sequence
    data.sort(key=lambda x: len(x[0]), reverse=True)

    # seperate source and target sequences
    src_seqs, trg_seqs, pos_seqs,lm_seqs,label = zip(*data)

    # merge sequences (from tuple of 1D tensor to 2D tensor)
    src_seqs, src_lengths = merge(src_seqs)
    trg_seqs, trg_lengths = merge(trg_seqs)
    pos_seqs, pos_lengths = merge(pos_seqs)
    # lm_seqs, lm_lengths = merge(lm_seqs)
    label = torch.tensor(label)
    if USE_CUDA:
        src_seqs = src_seqs.cuda()
        trg_seqs = trg_seqs.cuda()
        pos_seqs = pos_seqs.cuda()
        # lm_seqs = lm_seqs.cuda()
        label = label.cuda()
    return Variable(LongTensor(src_seqs)), Variable(LongTensor(trg_seqs)), Variable(LongTensor(pos_seqs)),lm_seqs, label

def collate_fn_keyword(data):
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    # sort a list by sequence length (descending order) to use pack_padded_sequence
    data.sort(key=lambda x: len(x[0]), reverse=True)

    # seperate source and target sequences
    src_seqs, trg_seqs, pos_seqs, lm_seqs, total_input_length, meta, keyword_x = zip(*data)

    # merge sequences (from tuple of 1D tensor to 2D tensor)
    src_seqs, src_lengths = merge(src_seqs)
    trg_seqs, trg_lengths = merge(trg_seqs)
    pos_seqs, pos_lengths = merge(pos_seqs)
    lm_seqs, lm_lengths = merge(lm_seqs)
    keyword_x, _ = merge(keyword_x)
    if USE_CUDA:
        src_seqs = src_seqs.cuda()
        trg_seqs = trg_seqs.cuda()
        pos_seqs = pos_seqs.cuda()
        lm_seqs = lm_seqs.cuda()
        keyword_x = keyword_x.cuda()
    return Variable(LongTensor(src_seqs)), Variable(LongTensor(trg_seqs)), Variable(LongTensor(pos_seqs)),Variable(LongTensor(lm_seqs)), total_input_length, meta,Variable(LongTensor(keyword_x))

class SnliDataset(Dataset):
    """Take a list of samples with form [[x,...],y,meta]
    """
    # need 3 special tokens
    # # as <ref start> 2
    # $ as <speaker1> 3
    # % as <speaker2> 4
    # '<|endoftext|>' as <eos> 50256
    def _split(self,data):
        positive_label = set(['entailment'])
        premise = []
        hypothesis = []
        label = []
        for p,h,l in tqdm(data):
            premise.append(self.tokenizer.encode(text_standardize(p)))
            hypothesis.append(self.tokenizer.encode(text_standardize(h)))
            if l in positive_label:
                label.append(torch.tensor(1))
            else:
                label.append(torch.tensor(0))
        return premise,hypothesis,label
    
    def _filter(self,premise,hypothesis,label,filter_mode=None):
        data = zip(premise,hypothesis,label)
        if filter_mode == None:
            data_filt = data
        else:
            data_filt = [x for x in data if x[2]!='-']
            
        premise_filt,hypothesis_filt,label_filt = zip(*data_filt)
        return premise_filt,hypothesis_filt,label_filt

    def parse_snli(self,path=None):
        with open(path) as f:
            data = [json.loads(line) for line in f]
        data_processed = [(line['sentence1'],line['sentence2'],line['gold_label']) for line in data]
        return data_processed

    def __init__(self,tokenizer,path='../data/snli_1.0/snli_1.0_train.jsonl',filter_mode=None,num_turns=5):
        
        self.data = self.parse_snli(path)
        self.tokenizer = tokenizer
        self.premise_encoded,self.hypothesis_encoded,self.label = self._split(self.data)
        self.premise_encoded,self.hypothesis_encoded,self.label = self._filter(self.premise_encoded,self.hypothesis_encoded,self.label,filter_mode)
        self.ref_start, self.speaker1,self.speaker2,self.eos = 2,3,4,50256

    def __getitem__(self,index):
        x = []
        type_x = []
        lm_x = []
        
        x += [self.speaker1]
        x += self.premise_encoded[index]
        type_x += [self.speaker1]*(len(self.premise_encoded[index])+1) # the premise part
        
        x += [self.ref_start] 
        x += self.hypothesis_encoded[index]
        x += [self.eos]
        type_x += [self.ref_start]*(len(self.hypothesis_encoded[index])+2) # the hypothesis part
        
        label = self.label[index]
        
        position_x = list(range(len(x)))

        x = torch.Tensor(x)
        type_x = torch.Tensor(type_x)
        position_x = torch.Tensor(position_x)
        
        return x,type_x,position_x,lm_x,label

    def __len__(self):
        return len(self.premise_encoded)

class GptDataset_full(Dataset):
    def _split(self,x_y_meta):
        x_all = []
        y_all = []
        meta_all = []
        aug_all = []
        keyword_all = []
        for x, y, meta, aug, keyword in x_y_meta:
            meta_all.append(meta)
            # update for the new data format
            aug = ''.join([a[1] for a in aug])
            x_all.append([self.tokenizer.encode(text_standardize(x_i)) for x_i in x])
            y_all.append(self.tokenizer.encode(text_standardize(y)))
            aug_all.append(self.tokenizer.encode(text_standardize(aug)))
            keyword_all.append(self.tokenizer.encode(text_standardize(keyword)))
        return x_all,y_all,meta_all,aug_all, keyword_all

    def _filt(self, length=1024):
        data = zip(self.x_encoded,self.y_encoded,self.meta,self.aug_encoded, self.keyword_encoded)
        data = [sample for sample in data if sum([len(sen) for sen in sample[0]][-self.args.num_turns:])+len(sample[1])+len(sample[3])+len(sample[4]) < 850]
        self.x_encoded,self.y_encoded,self.meta,self.aug_encoded, self.keyword_encoded = zip(*data)
        self.x_encoded = list(self.x_encoded)
        self.y_encoded = list(self.y_encoded)
        self.meta = list(self.meta)
        self.aug_encoded = list(self.aug_encoded)
        self.keyword_encoded = list(self.keyword_encoded)

    def __init__(self,x_y_meta,tokenizer,args):
        self.x_y_meta = x_y_meta
        self.num_turns = args.num_turns
        self.tokenizer = tokenizer
        self.args = args
        self.x_encoded,self.y_encoded,self.meta,self.aug_encoded, self.keyword_encoded = self._split(x_y_meta)
        self._filt() # TODO: add back filt for mix-review
        self.ref_start, self.speaker1,self.speaker2,self.eos = 2,3,4,50256
        self.augment = 5
        if self.args.augment:
            print("Using augment sentences.")
        if self.args.keyword:
            print("Using keywords.")

    def __getitem__(self,index):
        x = []
        type_x = []
        lm_x = []

        if self.args.augment:
            x += [self.augment] + self.aug_encoded[index]
        if self.args.keyword:
            x += [self.augment] + self.keyword_encoded[index]
        type_x += [self.augment] * len(x)

        is_speaker1 = bool(self.num_turns % 2) # which speaker start the conversation

        for utt in self.x_encoded[index][-self.num_turns:]:
            if is_speaker1: # add the prefix special token for each utterance
                x+=[self.speaker1]
                type_x += [self.speaker1]*(len(utt)+1)
            else:
                x+=[self.speaker2]
                type_x += [self.speaker2]*(len(utt)+1)
            x += utt
            is_speaker1 = not is_speaker1
        lm_x += [-100]*len(x) # all position for the input is masked for loss calculation

        total_input_length = len(x)

        x += [self.ref_start] + self.y_encoded[index] + [self.eos]

        type_x += [self.ref_start]*(len(self.y_encoded[index])+2)
        lm_x += [-100] + self.y_encoded[index] + [self.eos]
        position_x = list(range(len(x)))

        x = torch.Tensor(x)
        type_x = torch.Tensor(type_x)
        position_x = torch.Tensor(position_x)
        lm_x = torch.Tensor(lm_x)
        x_len = x.shape[0]

        return x,type_x,position_x,lm_x,total_input_length,self.meta[index]

    def __len__(self):
        return len(self.x_encoded)

class GptDataset_nli(GptDataset):
    def __init__(self, x_y_meta, tokenizer, filter_mode=None, num_turns=5):
        super(GptDataset_nli, self).__init__(x_y_meta, tokenizer)
        self.label = [1] * len(self.x_encoded) + [0] * len(self.x_encoded)
        self.x_encoded = self.x_encoded + self.x_encoded
        self.y_encoded = list(self.y_encoded) + random.sample(self.y_encoded, len(self.y_encoded))
        self.x_encoded, self.y_encoded, self.label = zip(
            *random.sample(list(zip(self.x_encoded, self.y_encoded, self.label)), len(self.x_encoded)))

    def __getitem__(self, index):
        # former utterances - premise -speaker1
        # response - hypothesis - ref_start
        x = []
        type_x = []
        lm_x = []
        is_speaker1 = bool(len(self.x_encoded[index]) % 2)  # which speaker start the conversation

        for utt in self.x_encoded[index]:
            if is_speaker1:  # add the prefix special token for each utterance
                x += [self.speaker1]
                type_x += [self.speaker1] * (len(utt) + 1)
            else:
                x += [self.speaker2]
                type_x += [self.speaker2] * (len(utt) + 1)
            x += utt
            is_speaker1 = not is_speaker1

        total_input_length = len(x)

        x += [self.ref_start] + self.y_encoded[index] + [self.eos]

        type_x += [self.ref_start] * (len(self.y_encoded[index]) + 2)
        position_x = list(range(len(x)))

        x = torch.Tensor(x)
        type_x = torch.Tensor(type_x)
        position_x = torch.Tensor(position_x)
        x_len = x.shape[0]

        return x, type_x, position_x, lm_x, self.label[index]

class GptDataset_full_condition(Dataset):
    def _split(self, x_y_meta):
        x_all = []
        y_all = []
        meta_all = []
        aug_all = []
        keyword_all = []
        for x, y, meta, aug, keyword in tqdm(x_y_meta):
            meta_all.append(meta)
            # update for the new data format
            aug = ''.join([a[1] for a in aug])
            x_all.append([self.tokenizer.encode(text_standardize(x_i)) for x_i in x])
            y_all.append(self.tokenizer.encode(text_standardize(y)))
            aug_all.append(self.tokenizer.encode(text_standardize(aug)))
            keyword_all.append(self.tokenizer.encode(text_standardize(keyword)))
        return x_all, y_all, meta_all, aug_all, keyword_all

    def _filt(self, length=1024):
        data = zip(self.x_encoded, self.y_encoded, self.meta, self.aug_encoded, self.keyword_encoded)
        data = [sample for sample in data if
                sum([len(sen) for sen in sample[0]][-self.args.num_turns:]) + len(sample[1]) + len(sample[3]) + len(
                    sample[4]) < 850]
        self.x_encoded, self.y_encoded, self.meta, self.aug_encoded, self.keyword_encoded = zip(*data)
        self.x_encoded = list(self.x_encoded)
        self.y_encoded = list(self.y_encoded)
        self.meta = list(self.meta)
        self.aug_encoded = list(self.aug_encoded)
        self.keyword_encoded = list(self.keyword_encoded)

    def __init__(self, x_y_meta, tokenizer, args):
        self.x_y_meta = x_y_meta
        self.num_turns = args.num_turns
        self.tokenizer = tokenizer
        self.args = args
        self.x_encoded, self.y_encoded, self.meta, self.aug_encoded, self.keyword_encoded = self._split(x_y_meta)
        self._filt()  # TODO: add back filt for mix-review

        self.ref, self.speaker1, self.speaker2 = tokenizer.ref, tokenizer.speaker1, tokenizer.speaker2
        self.eos = tokenizer.eos
        self.augment = tokenizer.augment

        self.code_set  =  set(['CR','SR', 'GIV', 'QUEST', 'SEEK', 'AF', 'EMPH', 'PWOP', 'PWP', 'CON'])
        for c in self.code_set:
            var_name = 'is_' + c.lower()
            setattr(self, var_name, getattr(tokenizer, var_name))
        print('Dataset initialized.')

    def __getitem__(self, index):
        x = []
        type_x = []
        lm_x = []

        is_speaker1 = bool(self.num_turns % 2)  # which speaker start the conversation

        code_token = getattr(self, 'is_' + self.meta[index][2].lower())
        x += [code_token]
        type_x += [code_token]
        # if self.meta[index][2] in self.code_set:
        #     x += [self.is_non_ref]
        #     type_x += [self.is_non_ref]
        # else:
        #     x += [self.is_ref]
        #     type_x += [self.is_ref]

        for utt in self.x_encoded[index][-self.num_turns:]:
            if is_speaker1:  # add the prefix special token for each utterance
                x += [self.speaker1]
                type_x += [self.speaker1] * (len(utt) + 1)
            else:
                x += [self.speaker2]
                type_x += [self.speaker2] * (len(utt) + 1)
            x += utt
            is_speaker1 = not is_speaker1
        lm_x += [-100] * len(x)  # all position for the input is masked for loss calculation

        total_input_length = len(x)

        x += [self.ref] + self.y_encoded[index] + [self.eos]

        type_x += [self.ref] * (len(self.y_encoded[index]) + 2)
        lm_x += [-100] + self.y_encoded[index] + [self.eos]
        position_x = list(range(len(x)))

        x = torch.tensor(x)
        type_x = torch.tensor(type_x)
        position_x = torch.tensor(position_x)
        lm_x = torch.tensor(lm_x)
        x_len = x.shape[0]

        return x, type_x, position_x, lm_x, total_input_length, self.meta[index]

    def __len__(self):
        return len(self.x_encoded)


class GptDataset_KBERT_old(Dataset):
    def get_comet_aug_deque(self, comet_data, num_turns=5):
        clause_dq = deque()
        for comet_in, comet_out in comet_data:
            if comet_out == "":
                continue
            loc = int(comet_in.split()[0])
            if loc >= (10 - num_turns):
                clause_dq.append((loc, comet_out))
        return clause_dq

    def __init__(self, x_y_meta, tokenizer, args):
        self.data = x_y_meta
        self.num_turns = args.num_turns
        self.tokenizer = tokenizer
        self.args = args
        self.ref_start, self.speaker1, self.speaker2, self.eos = 2, 3, 4, 50256
        self.augment = 5

        if self.args.augment:
            print("Using augment sentences.")
        if self.args.keyword:
            print("Using keywords.")

    def __getitem__(self, index):
        x = []
        type_x = []
        lm_x = []
        soft_position_x = []

        dq = self.get_comet_aug_deque(self.data[index][3])  # the comet info

        mask_info = []

        context = self.data[index][0]
        response = self.data[index][1]

        is_speaker1 = bool(self.args.num_turns % 2)
        soft_loc = 0  # keep tract of the location of main sentences, point to the next token to be added
        utterance_start_loc = 0
        for i in range(10 - self.args.num_turns, 10):
            utternace_encoded = self.tokenizer.encode(text_standardize(context[i]))

            # add the prefix special token for each utterance
            if is_speaker1:
                x += [self.speaker1]
                type_x += [self.speaker1] * (len(utternace_encoded) + 1)
            else:
                x += [self.speaker2]
                type_x += [self.speaker2] * (len(utternace_encoded) + 1)
            x += utternace_encoded
            utterance_end_loc = len(x)

            soft_position_x += list(range(soft_loc, soft_loc + len(utternace_encoded) + 1))

            # add the aug, if it is the right place
            while len(dq) != 0 and dq[0][0] == i:
                comet_output = dq.popleft()[1]
                comet_encoded = self.tokenizer.encode(text_standardize(comet_output))

                x += [self.augment] + comet_encoded
                type_x += [self.augment] * (len(comet_encoded) + 1)
                soft_position_x += list(range(soft_loc, soft_loc + len(comet_encoded) + 1))
                mask_info.append([utterance_start_loc, utterance_end_loc, len(comet_encoded)+1])
            # update the pointer to the new seq end, add one for the delimiter token
            soft_loc += len(utternace_encoded) + 1
            is_speaker1 = not is_speaker1
            utterance_start_loc = len(x)

        lm_x += [-100] * len(x)  # all position for the input is masked for loss calculation
        total_input_length = len(x)

        response_encoded = self.tokenizer.encode(text_standardize(response))
        x += [self.ref_start] + response_encoded + [self.eos]

        type_x += [self.ref_start] * (len(response_encoded) + 2)
        lm_x += [-100] + response_encoded + [self.eos]

        soft_position_x += list(range(soft_loc, soft_loc + len(response_encoded) + 2))

        x = torch.Tensor(x)
        type_x = torch.Tensor(type_x)
        soft_position_x = torch.Tensor(soft_position_x)
        lm_x = torch.Tensor(lm_x)
        x_len = x.shape[0]

        # process the mask
        attention_mask = torch.tril(torch.ones(x_len, x_len))
        for u_start, u_end, branch_len in mask_info:
            attention_mask[u_end+branch_len+1: u_end+1:u_end+branch_len+1] = 0 # [1st token after branch: , 1st token in branch: last token in branch+1]
        attention_mask = attention_mask.view(1, x_len, x_len)

        return x, type_x, soft_position_x, lm_x, total_input_length, attention_mask

    def __len__(self):
        return len(self.data)


class GptDataset_KBERT(Dataset):
    def __init__(self, tokenizer, args, file_path="../data_processed/data_comet_dict"):
        pickle_handler = open(file_path, 'rb')

        self.data = pickle.load(pickle_handler)
        
        self.max_length = 510
        self.tokenizer = tokenizer
        self.args = args
        self.num_turns = args.num_turns
        self.ref, self.speaker1, self.speaker2 = tokenizer.ref, tokenizer.speaker1, tokenizer.speaker2
        self.eos = tokenizer.eos
        self.augment = tokenizer.augment

        if not self.args.kbert:
            self.args.kbert_mask = False
            self.args.kbert_position = False
            print("Not using kbert scheme.")
        if self.args.kbert_mask:
            print("using kbert-style attention mask")
        if self.args.kbert_position:
            print("using kbert-style soft-postional encoding")

    def __getitem__(self, index):
        # preprare variables
        x = []
        type_x = []
        lm_x = []
        soft_position_x = []
        attention_mask = []

        # 0. unpack needed input info
        context = self.data[index]['context']
        srl_mask = self.data[index]['srl_mask']
        comet_output = self.data[index]['comet']  # a list of dict or None
        response = self.data[index]['response']

        # 1. encode the response.
        response_encoded = self.tokenizer.encode(text_standardize(response))

        # 2. encode each utterance.
        context_encoded = []
        for i in range(10 - self.args.num_turns, 10):
            context_encoded.append(self.tokenizer.encode(text_standardize(context[i])))

        # 3. encode the comet output for each utterance.
        comet_encoded = []
        for i in range(len(comet_output)):
            comet_text_i = ""
            if comet_output[i] is None:
                comet_encoded.append(None)
                continue
            for rel in comet_output[i]:
                for candidate in comet_output[i][rel]['beams']:
                    if candidate != 'none':
                        comet_text_i += rel + " " + candidate + " "
                        break
            comet_encoded.append(self.tokenizer.encode(text_standardize(comet_text_i)))

        # 4. use the encoded seq to build the input and attention mask
        is_speaker1 = bool(self.args.num_turns % 2)
        soft_loc = 0
        for i in range(self.args.num_turns):

            # add an utterance. update x & type_x
            if is_speaker1:
                x += [self.speaker1]
                type_x += [self.speaker1] * (len(context_encoded[i]) + 1)
            else:
                x += [self.speaker2]
                type_x += [self.speaker2] * (len(context_encoded[i]) + 1)
            x += context_encoded[i]

            # update pos_x
            # concate aug part after x. but the index is from the last related token
            soft_position_x += list(range(soft_loc, soft_loc + (len(context_encoded[i]) + 1)))

            last_related_token_index = len(srl_mask[i]) - 1 - srl_mask[i][::-1].index(1)

            # add comet output
            if self.args.kbert:
                if comet_encoded[i] is not None:
                    x += [self.augment] + comet_encoded[i]
                    type_x += [self.augment] * (len(comet_encoded[i]) + 1)

                    # +2 for the special token and the requirement of one-number larger than the utterance
                    soft_position_x += list(range(soft_loc + 2 + last_related_token_index,
                                                  soft_loc + 2 + last_related_token_index + (len(comet_encoded[i]) + 1)))

            soft_loc += (len(context_encoded[i]) + 1)
            is_speaker1 = not is_speaker1

        lm_x += [-100] * len(x)  # all position for the input is masked for loss calculation
        total_input_length = len(x)

        response_encoded = self.tokenizer.encode(text_standardize(response))
        x += [self.ref] + response_encoded + [self.eos]

        type_x += [self.ref] * (len(response_encoded) + 2)
        lm_x += [-100] + response_encoded + [self.eos]

        soft_position_x += list(range(soft_loc, soft_loc + len(response_encoded) + 2))
        
        
        x = x[:self.max_length]
        type_x = type_x[:self.max_length]
        lm_x = lm_x[:self.max_length]
        soft_position_x = soft_position_x[:self.max_length]
        
        # build attention mask
        attention_mask = torch.tril(torch.ones(len(x), len(x)))
        if self.args.kbert_mask:
            aug_start = 0  # where the aug begin
            utt_start = 0  # where the utt begin

            for turn in range(self.args.num_turns):
                aug_start += len(context_encoded[turn]) + 1
                # iter through every token in the comet output
                if comet_encoded[turn] is not None:
                    for aug_token_pos in range(aug_start, aug_start + len(comet_encoded[turn]) + 1):
                        # set the attention related to the aug part to be all zero
                        attention_mask[aug_token_pos, :] = torch.zeros_like(attention_mask[aug_token_pos, :])

                        attention_mask[:, aug_token_pos] = torch.zeros_like(attention_mask[:, aug_token_pos])
                        # set attention on related token to be one
                        for normal_token_pos in range(len(context_encoded[turn])):
                            attention_mask[aug_token_pos, utt_start + normal_token_pos + 1] += srl_mask[turn][
                                normal_token_pos]
                        # set attention on previous aug tokens to be one
                        for previous_aug_token_poc in range(aug_start, aug_token_pos + 1):
                            attention_mask[aug_token_pos, previous_aug_token_poc] += 1

                    aug_start += len(comet_encoded[turn]) + 1
                    utt_start += len(comet_encoded[turn]) + 1
                utt_start += (len(context_encoded[turn]) + 1)

        x = torch.tensor(x)
        type_x = torch.tensor(type_x)
        if not self.args.kbert_position:
            soft_position_x = list(range(len(x)))
        soft_position_x = torch.tensor(soft_position_x)
        lm_x = torch.tensor(lm_x)
        return x, type_x, soft_position_x, lm_x, total_input_length, attention_mask

    def __len__(self):
        return len(self.data)

def get_data(args, tokenizer, split_size):
    """
    Return the data loaders needed for training and evaluation.
    :param args: command line arguments.
    :param tokenizer: the tokenizer used in preparing the data.
    :param split_size: the portion of train, test, validation set.
    :return data_loader: The data loader for the training set.
    :return val_loader: The data loader for the validation set.
    """
    # random.seed(args.seed)
    # torch.random.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    # torch.manual_seed(args.seed)
    if args.special_input:
        print("Using mutated data.")
        pickle_handler = open('../data_processed/' + args.special_input, 'rb')
        x_y_meta = pickle.load(pickle_handler)
        gpt_data = GptDataset(x_y_meta, tokenizer, args.output_dir, num_turns=args.num_turns)
        # TODO: not finished
#     #======================origin without kbert======
#     elif not args.kbert:
#         print("Using full data.")
#         pickle_handler = open('../data_processed/x_y_with_comet', 'rb') # TODO: change back to the old data.
#         x_y_meta = pickle.load(pickle_handler)
#         gpt_data = GptDataset_full(x_y_meta, tokenizer, args=args)
    elif args.conditional:
        print("using conditional generation data")
        file_path = "../data_processed/0423_"
        # data_train = pickle.load(open(file_path+'train_ref', 'rb')) + pickle.load(open(file_path+'train_non_ref','rb'))
        data_train = pickle.load(open(file_path + 'train', 'rb'))[:1000]
        gpt_train= GptDataset_full_condition(data_train, tokenizer, args=args)

        data_test = pickle.load(open(file_path+'test', 'rb'))[:100]
        gpt_test = GptDataset_full_condition(data_test, tokenizer, args=args)

        data_val = pickle.load(open(file_path+'test', 'rb'))[:100]
        gpt_val = GptDataset_full_condition(data_val, tokenizer, args=args)
    elif args.kbert:
        print("Using KBERT data")
        gpt_data = GptDataset_KBERT(tokenizer, args=args)
        print("Dataset initialized. There are {} samples.".format(len(gpt_data)))

        test_size = int(len(gpt_data) * split_size['test'])
        val_size = int(len(gpt_data) * split_size['val'])

        gpt_train, gpt_test, gpt_val = torch.utils.data.random_split(gpt_data,
                                                                     [len(gpt_data) - test_size - val_size, test_size,
                                                                      val_size])
    else:
        print("Using full data.")
        file_path = "../data_processed/"
        data_train = pickle.load(open(file_path+'train_ref', 'rb'))
        gpt_train= GptDataset_full_condition(data_train[:], tokenizer, args=args)

        data_test = pickle.load(open(file_path+'test_ref', 'rb'))
        gpt_test = GptDataset_full_condition(data_test[:], tokenizer, args=args)

        data_val = pickle.load(open(file_path+'test_ref', 'rb'))
        gpt_val = GptDataset_full_condition(data_val[: ], tokenizer, args=args)

        # # ======= one-time plug in========
        # x_y_meta_pre = pickle.load(open("../data_processed/data_comet_dict",'rb'))
        # x_y_meta = []
        # for x in x_y_meta_pre:
        #     context_i = x['context']
        #     response_i = x['response']
        #     meta_i = x['meta']
        #     x_y_meta.append([context_i, response_i, meta_i, "", ""])
        #
        # with open('../data_processed/train_ref', 'wb') as f:
        #     idx = gpt_train.indices
        #     gpt_train =[x_y_meta[i] for i in idx]
        #     pickle.dump(gpt_train, f)
        # with open('../data_processed/test_ref', 'wb') as f:
        #     idx = gpt_test.indices
        #     gpt_test = [x_y_meta[i] for i in idx]
        #     pickle.dump(gpt_test, f)
        # with open('../data_processed/val_ref', 'wb') as f:
        #     idx = gpt_val.indices
        #     gpt_val = [x_y_meta[i] for i in idx]
        #     pickle.dump(gpt_val, f)

    if 'train_batch_size' not in args:
        args.train_batch_size = 1
    if args.kbert:
        print("using kbert collate_fn")
        data_loader = DataLoader(dataset=gpt_train, batch_size=args.train_batch_size, shuffle=True, drop_last=True,
                                collate_fn=collate_fn)
        test_loader = DataLoader(dataset=gpt_test, batch_size=1, shuffle=False, drop_last=False, collate_fn=collate_fn)
        val_loader = DataLoader(dataset=gpt_val, batch_size=1, shuffle=False, drop_last=False, collate_fn=collate_fn)
    else:
        print("using normal collate_fn")
        data_loader = DataLoader(dataset=gpt_train, batch_size=args.train_batch_size, shuffle=True, drop_last=True,
                                collate_fn=collate_fn_conditional)
        test_loader = DataLoader(dataset=gpt_test, batch_size=1, shuffle=False, drop_last=False, collate_fn=collate_fn_conditional)
        val_loader = DataLoader(dataset=gpt_val, batch_size=1, shuffle=False, drop_last=False, collate_fn=collate_fn_conditional)
    return data_loader, test_loader, val_loader

def prepare_mix_review(args, tokenizer):
    print("Preparing Alexander dataset")
    pickle_handler = open('../data_processed/data_alex', 'rb')
    data = pickle.load(pickle_handler)
    gpt_alex = GptDataset_full(data, tokenizer, args=args)
    print("Alexander dataset prepared. Has {} samples".format(len(gpt_alex)))
    return gpt_alex

def update_mix_review(gpt_train, gpt_alex, epoch, args, mix_ratio=4, mix_decay=0.7, collate_fn=collate_fn):
    mix_amount = int(mix_ratio*(0.7**epoch)*len(gpt_train))
    gpt_alex_active,_ = torch.utils.data.random_split(gpt_alex, [mix_amount, len(gpt_alex)-mix_amount])

    data_loader = DataLoader(dataset=gpt_train+gpt_alex_active, batch_size=args.train_batch_size, shuffle=True, drop_last=True,
                                collate_fn=collate_fn)
    return data_loader