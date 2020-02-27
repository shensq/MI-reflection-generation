import torch
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable
import json
import random
import sys
import pickle
from tqdm import tqdm
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

# class GptDataset_keyword(Dataset):
#     def _split(self,x_y_meta):
#         x_all = []
#         y_all = []
#         meta_all = []
#         aug_all = []
#         for x,y,meta,aug in x_y_meta:
#             meta_all.append(meta)
#             x_all.append([self.tokenizer.encode(text_standardize(x_i)) for x_i in x])
#             y_all.append(self.tokenizer.encode(text_standardize(y)))
#             key_word.append(self.tokenizer.encode(text_standardize(aug)))

#         return x_all,y_all,meta_all,aug_all

#     def __init__(self,x_y_meta,tokenizer,num_turns=5):
#         self.x_y_meta = x_y_meta
#         self.num_turns = num_turns
#         self.tokenizer = tokenizer
#         self.x_encoded,self.y_encoded,self.meta,self.aug_encoded = self._split(x_y_meta)
#         self.ref_start, self.speaker1,self.speaker2,self.eos = 2,3,4,50256
#         self.augment = 5
#         self.keyword = 10 # '+'

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

    # sort a list by sequence length (descending order) to use pack_padded_sequence
    data.sort(key=lambda x: len(x[0]), reverse=True)

    # seperate source and target sequences
    src_seqs, trg_seqs, pos_seqs,lm_seqs,total_input_length,meta = zip(*data)

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

class GptDataset_keyword(Dataset):
    def _split(self, x_y_meta):
        x_all = []
        y_all = []
        meta_all = []
        keyword_all = []
        for x, y, meta, keyword in x_y_meta:
            meta_all.append(meta)
            x_all.append([self.tokenizer.encode(text_standardize(x_i)) for x_i in x])
            y_all.append(self.tokenizer.encode(text_standardize(y)))
            keyword_all.append(self.tokenizer.encode(text_standardize(keyword)))
        return x_all, y_all, meta_all, keyword_all

    def __init__(self, x_y_meta, tokenizer, num_turns=5):

        self.x_y_meta = x_y_meta
        self.num_turns = num_turns
        self.tokenizer = tokenizer
        self.x_encoded, self.y_encoded, self.meta, self.keyword = self._split(x_y_meta)
        self.ref_start, self.speaker1, self.speaker2, self.eos = 2, 3, 4, 50256

    def __getitem__(self, index):
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
        lm_x += [-1] * len(x)  # all position for the input is masked for loss calculation

        total_input_length = len(x)

        x += [self.ref_start] + self.y_encoded[index] + [self.eos]

        type_x += [self.ref_start] * (len(self.y_encoded[index]) + 2)
        lm_x += [-1] + self.y_encoded[index] + [self.eos]
        position_x = list(range(len(x)))

        x = torch.Tensor(x)
        type_x = torch.Tensor(type_x)
        position_x = torch.Tensor(position_x)
        lm_x = torch.Tensor(lm_x)
        x_len = x.shape[0]

        keyword_x = [] + self.keyword[index]
        keyword_x = torch.Tensor(keyword_x)
        return x, type_x, position_x, lm_x, total_input_length, self.meta[index], keyword_x

    def __len__(self):
        return len(self.x_encoded)

# class GptDataset_nli(GptDataset):
#     def __init__(self, x_y_meta, tokenizer, filter_mode=None,num_turns=5,augment=True):
#         super(GptDataset_nli, self).__init__(x_y_meta,tokenizer, num_turns=num_turns)
#         self.augment = augment
#         self.pos_len = len(self.x_encoded)

#     def __len__(self):
#         if self.augment:
#             return 2 * len(self.x_encoded)
#         else:
#             return len(self.x_encoded)

#     def __getitem__(self,index):
#         # client utterances - premise -speaker1 
#         # response - hypothesis - ref_start
#         true_index = index
#         if index >= self.pos_len:
#             index = index - self.pos_len

#         x = []
#         type_x = []
#         lm_x = []
#         is_speaker1 = bool(len(self.x_encoded[index])%2) # which speaker start the conversation
        
#         x+=[self.speaker1]
#         type_x += [self.speaker1]
#         for utt in self.x_encoded[index][-self.num_turns:]:
#             if is_speaker1: # add the prefix special token for each utterance
#                 type_x += [self.speaker1]*(len(utt))
#                 x += utt
#             # else:
#             #     x+=[self.speaker2]
#             #     type_x += [self.speaker2]*(len(utt)+1)
#             #     x += utt
#             is_speaker1 = not is_speaker1

#         total_input_length = len(x)
        
#         if true_index >= self.pos_len:
#             rand_index = random.randint(0,self.pos_len-1)
#             x += [self.ref_start] + self.y_encoded[rand_index] + [self.eos]
#             type_x += [self.ref_start]*(len(self.y_encoded[rand_index])+2)
#         else:
#             x += [self.ref_start] + self.y_encoded[index] + [self.eos]
#             type_x += [self.ref_start]*(len(self.y_encoded[index])+2)
#         position_x = list(range(len(x)))

#         x = torch.Tensor(x)
#         type_x = torch.Tensor(type_x)
#         position_x = torch.Tensor(position_x)
#         x_len = x.shape[0]
#         label = torch.tensor(0) if true_index>self.pos_len else torch.tensor(1)
#         return x,type_x,position_x,lm_x, label

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


class GptDataset_nli(GptDataset_full):
    def __init__(self, x_y_meta, tokenizer, args, infer=False):
        super(GptDataset_nli, self).__init__(x_y_meta,tokenizer, args)
        self.pos_len = len(self.x_encoded)
        self.num_turns = 5
        self.infer = infer
    def __len__(self):
        if self.infer:
            return len(self.x_encoded)
        else:
            return 2 * len(self.x_encoded)

    def __getitem__(self,index):
        # client utterances - premise -speaker1 
        # response - hypothesis - ref_start
        true_index = index
        if index >= self.pos_len:
            index = index - self.pos_len

        x = []
        type_x = []
        lm_x = []
        is_speaker1 = bool(len(self.x_encoded[index])%2) # which speaker start the conversation
        
        x+=[self.speaker1]
        type_x += [self.speaker1]
        for utt in self.x_encoded[index][-self.num_turns:]:
            if is_speaker1: # add the prefix special token for each utterance
                type_x += [self.speaker1]*(len(utt))
                x += utt
            # else:
            #     x+=[self.speaker2]
            #     type_x += [self.speaker2]*(len(utt)+1)
            #     x += utt
            is_speaker1 = not is_speaker1

        total_input_length = len(x)
        
        if true_index >= self.pos_len:
            rand_index = random.randint(0,self.pos_len-1)
            x += [self.ref_start] + self.y_encoded[rand_index] + [self.eos]
            type_x += [self.ref_start]*(len(self.y_encoded[rand_index])+2)
        else:
            x += [self.ref_start] + self.y_encoded[index] + [self.eos]
            type_x += [self.ref_start]*(len(self.y_encoded[index])+2)
        position_x = list(range(len(x)))

        x = torch.Tensor(x)
        type_x = torch.Tensor(type_x)
        position_x = torch.Tensor(position_x)
        x_len = x.shape[0]
        label = torch.tensor(0) if true_index>self.pos_len else torch.tensor(1)

        return x,type_x,position_x,lm_x, label

class XLDataset_nli(GptDataset_nli):
    def __init__(self, x_y_meta, tokenizer, args, infer=False):
        super(GptDataset_nli, self).__init__(x_y_meta,tokenizer, args)
        self.pos_len = len(self.x_encoded)
        self.num_turns = 5
        self.infer = infer
#         self.ref_start, self.speaker1,self.speaker2,self.eos = 2,3,4,50256
        self.pad, self.sep, self.cls = 5, 4, 3
        self.unk, self.s, self.s_bar = 0, 1, 2

    def __len__(self):
        if self.infer:
            return len(self.x_encoded)
        else:
            return 2 * len(self.x_encoded)

    def __getitem__(self,index):
        # client utterances - premise -speaker1 
        # response - hypothesis - ref_start
        true_index = index
        if index >= self.pos_len:
            index = index - self.pos_len
    
        
        x = []
        type_x = []
        lm_x = []
        mask_x = []
        is_speaker1 = bool(len(self.x_encoded[index])%2) # which speaker start the conversation
        
        
        for utt in self.x_encoded[index][-self.num_turns:]:
            if is_speaker1: # add the prefix special token for each utterance
                type_x += [self.unk]*(len(utt))
                x += utt
            else:
                type_x += [self.unk]*(len(utt))
                x += utt
            is_speaker1 = not is_speaker1
#         import pdb;pdb.set_trace()
        x += [self.sep]
        type_x += [self.unk]
        
        total_input_length = len(x)
        
        if true_index >= self.pos_len:
            rand_index = random.randint(0,self.pos_len-1)
            x += self.y_encoded[rand_index] + [self.sep, self.cls]
            type_x += [self.s]*(len(self.y_encoded[rand_index])+1) + [self.s_bar]
        else:
#             x += [self.ref_start] + self.y_encoded[index] + [self.eos]
#             type_x += [self.ref_start]*(len(self.y_encoded[index])+2)
            x += self.y_encoded[index] + [self.sep, self.cls]
            type_x += [self.s]*(len(self.y_encoded[index])+1) + [self.s_bar]
        
        position_x = list(range(len(x)))
        mask_x = [self.s] * len(x)
        
#         ####
#         x = x[-100:]
#         mask_x = mask_x[-100:]
#         type_x = type_x[-100:]
        
        # left padding 
        x = [self.pad] * (self.args.max_length-len(x)) + x[-self.args.max_length:]
        mask_x = [self.unk] * (self.args.max_length-len(mask_x)) + mask_x[-self.args.max_length:]
        type_x = [self.sep] * (self.args.max_length-len(type_x)) + type_x[-self.args.max_length:]
        
        x = torch.Tensor(x).long()
        mask_x = torch.Tensor(mask_x).long()
        type_x = torch.Tensor(type_x).long()
        position_x = torch.Tensor(position_x)
        
        x_len = x.shape[0]
        
        label = torch.tensor(0) if true_index>self.pos_len else torch.tensor(1)
        # label = torch.tensor(0) if true_index>self.pos_len else torch.tensor(0)
        
        if USE_CUDA:
            x = x.cuda()
            mask_x = mask_x.cuda()
            type_x = type_x.cuda()
            label = label.cuda()
#         return x, mask_x, type_x, label, position_x, lm_x
        return x, mask_x, type_x, label





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
    else:
        print("Using full data.")
        pickle_handler = open('../data_processed/x_y_meta_all', 'rb') # TODO: change back to the old data.
        x_y_meta = pickle.load(pickle_handler)
        gpt_data = GptDataset_full(x_y_meta, tokenizer, args=args)
    print("Dataset initialized. There are {} samples.".format(len(gpt_data)))

    test_size = int(len(gpt_data) * split_size['test'])
    val_size = int(len(gpt_data) * split_size['val'])

    gpt_train, gpt_test, gpt_val = torch.utils.data.random_split(gpt_data,
                                                                 [len(gpt_data) - test_size - val_size, test_size,
                                                                  val_size])

    # import pdb;pdb.set_trace()
    # with open('../../mi_data/train','wb') as f:
    #     pickle.dump(gpt_train, f)
    # with open('../../mi_data/test','wb') as f:
    #     pickle.dump(gpt_test, f)
    # with open('../../mi_data/val','wb') as f:
    #     pickle.dump(gpt_val, f)

    if 'train_batch_size' not in args:
        args.train_batch_size = 1
    data_loader = DataLoader(dataset=gpt_train, batch_size=args.train_batch_size, shuffle=True, drop_last=True,
                            collate_fn=collate_fn)
    test_loader = DataLoader(dataset=gpt_test, batch_size=1, shuffle=False, drop_last=False, collate_fn=collate_fn)
    val_loader = DataLoader(dataset=gpt_val, batch_size=1, shuffle=False, drop_last=False, collate_fn=collate_fn)
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

def get_data_old(args, tokenizer, split_size):
    """
    Return the data loaders needed for training and evaluation.
    :param args: command line arguments.
    :param tokenizer: the tokenizer used in preparing the data.
    :param split_size: the portion of train, test, validation set.
    :return data_loader: The data loader for the training set.
    :return val_loader: The data loader for the validation set.
    """
    if args.special_input:
        print("Using mutated data.")
        pickle_handler = open('../data_processed/' + args.special_input, 'rb')
        x_y_meta = pickle.load(pickle_handler)
        if args.augment:
            print("testing keywords with augment loader.")
            gpt_data = GptDataset_aug(x_y_meta, tokenizer, num_turns=args.num_turns)
        else:
            gpt_data = GptDataset(x_y_meta, tokenizer, args.output_dir, num_turns=args.num_turns)
    elif args.augment:
        print("Using augmented data")
        pickle_handler = open('../data_processed/x_y_meta_aug', 'rb')
        x_y_meta = pickle.load(pickle_handler)
        gpt_data = GptDataset_aug(x_y_meta, tokenizer, num_turns=args.num_turns)
    elif args.keyword:
        print("Using keyword cross attention")
        pickle_handler = open('../data_processed/x_y_meta_keyword', 'rb')
        x_y_meta = pickle.load(pickle_handler)
        gpt_data = GptDataset_keyword(x_y_meta, tokenizer)
    else:
        print("Using vanilla data.")
        pickle_handler = open('../data_processed/x_y_meta', 'rb')
        x_y_meta = pickle.load(pickle_handler)
        gpt_data = GptDataset(x_y_meta, tokenizer, args.output_dir, num_turns=args.num_turns)

    print("Dataset initialized. There are {} samples.".format(len(gpt_data)))
    test_size = int(len(gpt_data) * split_size['test'])
    val_size = int(len(gpt_data) * split_size['val'])
    gpt_train, gpt_test, gpt_val = torch.utils.data.random_split(gpt_data,
                                                                 [len(gpt_data) - test_size - val_size, test_size,
                                                                  val_size])
    if args.keyword:
        data_loader = DataLoader(dataset=gpt_train, batch_size=args.train_batch_size, shuffle=True, drop_last=True,
                                 collate_fn=collate_fn_keyword)
        val_loader = DataLoader(dataset=gpt_val, batch_size=1, shuffle=False, drop_last=False,
                                collate_fn=collate_fn_keyword)
    else:
        data_loader = DataLoader(dataset=gpt_train, batch_size=args.train_batch_size, shuffle=True, drop_last=True,
                                 collate_fn=collate_fn)
        val_loader = DataLoader(dataset=gpt_val, batch_size=1, shuffle=False, drop_last=False, collate_fn=collate_fn)
    return data_loader, val_loader
