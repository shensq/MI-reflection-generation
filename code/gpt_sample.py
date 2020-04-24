#!/usr/bin/env python3
import sys
# sys.path.insert(0,'/home/shensq/LIT/pip_package') # make sure the modified version of pytorch_transformer
import transformers
# assert pytorch_transformers.__file__[-36:]=='pip_package/transformers/__init__.py'
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import argparse
import logging
import pickle
import re
from tqdm import trange
import random 
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable
from tqdm import tqdm, trange
from rouge import Rouge 
from utils import clean_text,text_standardize,values_lexicon_encode
from gpt_loader import GptDataset, collate_fn, GptDataset_aug, get_data
# import nltk
# from nltk.translate.meteor_score import meteor_score

def top_k_logits(logits, k):
    """
    Masks everything but the k top entries as -infinity (1e10).
    Used to mask logits such that e^-infinity -> 0 won't contribute to the
    sum of the denominator.
    """
    if k == 0:
        return logits
    else:
        values = torch.topk(logits, k)[0]
        batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
        return torch.where(logits < batch_mins, torch.ones_like(logits) * -1e10, logits)

def get_topic_keywords(meta):
    # TODO: temperary function
    keywords_up = []
    keywords_down = []
    if meta[1]=='Weight management':
        keywords_up += [6551, 4483, 2057, 9799, 4425, 4461, 4255, 5517]
        keywords_down += [46040, 21856, 2526, 13230, 7523, 15220]
    if meta[1]=='Smoking cessation':
        keywords_up += [46040, 21856, 2526, 13230, 7523, 15220]
        keywords_down += [6551, 4483, 2057, 9799, 4425, 4461, 4255, 5517]
    return keywords_up, keywords_down

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

def sample_sequence(model, length, context, num_samples=1, temperature=1,
                        top_k=0, top_p=0.0, device='cuda', attention_mask=None,args=None):
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0).repeat(num_samples, 1)
    generated = context
    prev = context
    past = None
    if args.kbert:
        attention_size = attention_mask.shape[-1]
        output_attention_mask = torch.tril(torch.ones(512, 512, dtype=attention_mask.dtype))
        output_attention_mask = output_attention_mask.view(1,1,*output_attention_mask.shape)
        output_attention_mask[:,:,:attention_size,:attention_size] = attention_mask
        if torch.cuda.is_available():
            output_attention_mask = output_attention_mask.cuda()
    with torch.no_grad():
        for i in range(length):
#             inputs = {'input_ids': generated, 'past': None, 'key_word': key_word, 'use_keyword':use_keyword}
            current_length = generated.shape[-1]
            if args.kbert:
                inputs = {'input_ids': generated, 'past': None, 'attention_mask':output_attention_mask[:,:,:current_length,:current_length]}
            else:
                inputs = {'input_ids': generated, 'past': None}
            logits, past = model(**inputs)
            next_token_logits = logits[0, -1, :] / (temperature if temperature>0 else 1.)
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            # if top_k > 0 or top_p > 0.0: # greedy, top_p, top_k
            if temperature == 0:
                next_token = torch.argmax(filtered_logits, dim=-1).unsqueeze(-1)
            else: # temperature
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            while (i == 0) and (next_token[0] == 50256):
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
            prev = next_token.unsqueeze(0)
            if next_token[0] in [50256]:
                break
    return generated



def load_model_data(args):
    #  === prepare data and model
    # ====== Load GPT2 model ========
    model_dir = '../models/'+args.model_dir
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    if USE_CUDA:
        model.cuda()
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    tokenizer.eos = 50256
    tokenizer.speaker1 = 50257
    tokenizer.speaker2 = 50258
    tokenizer.augment = 50259
    tokenizer.ref = 50260
    tokenizer.is_cr = 50261
    tokenizer.is_sr = 50262
    tokenizer.is_giv = 50263
    tokenizer.is_quest = 50264
    tokenizer.is_seek = 50265
    tokenizer.is_af = 50266
    tokenizer.is_emph = 50267
    tokenizer.is_pwop = 50268
    tokenizer.is_pwp = 50269
    tokenizer.is_con = 50270
    return model, tokenizer

def code_conditional_generation(args, sample, tokenizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x, type_x, pos_x, lm_x, x_len, attention_mask = sample
    input_len = x_len[0]  # The number of tokens of the context utterances
    context_tokens = x[0][:input_len + 1]  # at evaluation stage, the input is without the ground truth
    code_set  =  {'CR','SR', 'GIV', 'QUEST', 'SEEK', 'AF', 'EMPH', 'PWOP', 'PWP', 'CON'}
    for code in code_set:
        code_token = getattr(tokenizer, 'is_'+code.lower())
        x[0][0] = code_token
        type_x[0][0] = code_token

        for i in range(args.nsamples // args.batch_size):
            decode_length = min(int(0.5 * len(context_tokens)), 192)
            # if args.augment:
            #     decode_length = int(0.5 * (5/6) * len(context_tokens))
            out = sample_sequence(
                model=model, length=decode_length,
                context=context_tokens,
                temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                device=device, attention_mask=attention_mask, args=args
            )
            out = out[:, len(context_tokens):-1].tolist()  # the generated result,get rid of eos
            print(tokenizer.decode(x[0].tolist()[:len(context_tokens)]))
            print('/n')
            print(tokenizer.decode(out[0]))

def run_model(args, model, tokenizer, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    if args.length == -1:
        args.length = model.config.n_ctx // 2
    elif args.length > model.config.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % model.config.n_ctx)

    hyp = []
    ref = []
    context = []
#     f = open('../result/'+args.output_dir+'.txt','w')
#     f_ref = open('../result/reference_'+args.output_dir+'.txt','w')

    for sample in tqdm(test_loader):
        code_conditional_generation(args, sample, tokenizer)
        continue

        x, type_x, pos_x, lm_x, x_len, attention_mask = sample
        input_len = x_len[0] # The number of tokens of the context utterances
        context_tokens = x[0][:input_len+1] # at evaluation stage, the input is without the ground truth

        generated = 0
        for i in range(args.nsamples // args.batch_size):
            decode_length = min(int(0.5 * len(context_tokens)),192)
            # if args.augment:
            #     decode_length = int(0.5 * (5/6) * len(context_tokens))
            out = sample_sequence(
                model=model,length=decode_length,
                context=context_tokens,
                temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                device=device, attention_mask = attention_mask, args=args
            )           
            out = out[:, len(context_tokens):-1].tolist() # the generated result,get rid of eos

            ref.append(tokenizer.decode(x[0].tolist()[len(context_tokens):-1]))
#             f_ref.write(tokenizer.decode(x[0].tolist()[len(context_tokens):-1]))
#             f_ref.write('\n')

            hyp.append(tokenizer.decode(out[0]))
#             f.write(tokenizer.decode(out[0]))
#             f.write('\n')

            context.append(tokenizer.decode(x[0].tolist()[:len(context_tokens)]))
#     f.close()
#     f_ref.close()
    return hyp, ref, context

def print_metric(hyp, ref, context, effective_length=1024):
    # ===== Calculate rouge ========
    rouge = Rouge()
    print(len(hyp))
    print(len(ref))
    hyp, ref = zip(*[(x,y) for x,y in zip(hyp, ref) if len(x)>3 and len(y)>3])
    print(len(hyp))
    hyp = [x[:effective_length] for x in hyp]
    ref = [x[:effective_length] for x in ref]
    scores = rouge.get_scores(hyp, ref,avg=True)
    print("ROUGE",scores)

        
def calculate_metric(hyp, ref, context, effective_length=1024):
    # ===== Calculate rouge ========
    with open('../result/rouge.txt','a') as f_result:
        rouge = Rouge()
        print(len(hyp))
        print(len(ref))
        hyp, ref = zip(*[(x,y) for x,y in zip(hyp, ref) if len(x)>3 and len(y)>3])
        print(len(hyp))
        hyp = [x[:effective_length] for x in hyp]
        ref = [x[:effective_length] for x in ref]
        scores = rouge.get_scores(hyp, ref,avg=True)
        print("ROUGE",scores)
        import time 
        f_result.write(time.asctime()+'\n')
        f_result.write(args.model_dir+ '\t' + str(effective_length) +'\n')
        f_result.write(str(scores))
        f_result.write('\n')
    # == dump output====
    print("#ref{} #hyp{}".format(len(ref),len(hyp)))
    with open("../data_processed/output_" + args.model_dir+'p{}k{}'.format(args.top_p,args.top_k),'wb') as f_output:
        pickle.dump(zip(hyp,ref,context), f_output)
        
#     # ====== Calculate Meteor =========
#     meteor_sum = 0
#     for i in range(min(len(ref),len(hyp))):
#         meteor_sum += meteor_score([ref[i]],hyp[i])

#     meteor_sum/=min(len(ref),len(hyp))
#     print(meteor_sum)   

def rouge_rank(hyp, ref, context):
    rouge = Rouge()
    # import pdb;pdb.set_trace()
    hyp, ref = zip(*[(x,y) for x,y in zip(hyp, ref) if len(x)>3 and len(y)>3])
    scores = rouge.get_scores(hyp, ref,avg=False) # type: list
    scores_content = zip(scores, hyp, ref, context, range(len(hyp)))
    scores_content = sorted(scores_content, key=lambda x:x[0]['rouge-1']['f'], reverse=True)
    return scores_content

def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    
if __name__ == '__main__':
    USE_CUDA = torch.cuda.is_available()
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)
    logger = logging.getLogger(__name__)

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='345M_Alex', help='pretrained model name or path to local checkpoint')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nsamples", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=-1)
    parser.add_argument("--length", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0)
    parser.add_argument('--output_dir',type=str,default='generate', help="The name of the output file.")
    parser.add_argument('--modified_decoding', action='store_true')
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--special_input',type=str)
    parser.add_argument('--keyword', action='store_true')
    parser.add_argument('--kbert', action='store_true')
    parser.add_argument('--cross_attention', action='store_true')
    parser.add_argument('--num_turns', type=int, default=5)
    parser.add_argument('--kbert_mask', action='store_true')
    parser.add_argument('--kbert_position', action='store_true')
    parser.add_argument('--conditional', action='store_true')
    args = parser.parse_args()
    if args.batch_size == -1:
        args.batch_size = 1
    assert args.nsamples % args.batch_size == 0
    print(args)

    # Setup the random seeds.
    set_seed(args.seed)

    model, tokenizer = load_model_data(args)
    split_size = {'train': 0.90, 'test': 0.05, 'val': 0.05}
    data_loader, test_loader, val_loader = get_data(args, split_size=split_size, tokenizer=tokenizer)
    # model, tokenizer, test_loader = load_model_data(args) # TODO: this is for old get_data
    
#     seed_list = [0,10,]
#     seed_list = [20,30]
#     seed_list = [0,]
    seed_list = [args.seed]
    hyp_all = []
    ref_all = []
    context_all = []
    
    for seed in seed_list:
        set_seed(seed)
        print("Using random seed {}".format(seed))
        hyp, ref, context = run_model(args, model, tokenizer, test_loader)
        hyp_all += hyp
        ref_all += ref
        context_all += context
    sample_ranked = rouge_rank(hyp, ref, context)
    with open("../data_processed/rouge_rank_" + args.model_dir,'wb') as f:
        pickle.dump(sample_ranked, f)
    calculate_metric(hyp, ref, context)
    # calculate_metric(hyp, ref, context, 5)



