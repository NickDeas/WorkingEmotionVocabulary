import argparse
import os
import torch

import random
import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

from bs4 import BeautifulSoup
import re

import pandas as pd
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend

from torch.utils.data import Dataset, DataLoader

import json
from datetime import datetime

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
np.random.seed(SEED)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

INSTRUCTIONS = {
    'en': 'Write a sample response from someone that is feeling "{affective_state}" to the following fictional user message. The sample response should clearly express that the author feels {affective_state}. Do not, however, explicitly mention "{affective_state}" or use asterisks to indicate actions.',
    'es': 'Write a sample response from someone that is feeling "{affective_state}" to the following fictional user message. The sample response should clearly express that the author feels {affective_state}. Do not, however, explicitly mention "{affective_state}" or use asterisks to indicate actions. Respond only in Spanish.'
}

FP_TEMPLATE = '{language}_expression_masive_full.csv'

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run affective state expression experiments."
    )
    
    parser.add_argument('-cf', '--config',
                        type = str)
    
    parser.add_argument('-r', '--run-id',
                        type = str)
    
    parser.add_argument('-dd', '--data-dir',
                        type = str)
    
    parser.add_argument('-rd', '--results-dir',
                        type = str)
    
    parser.add_argument('-m', "--model", 
                        type=str, 
                        help="Model name (e.g., llama-2-7b, mistral-7b)")
    
    parser.add_argument('-l', "--language", 
                        type=str, 
                        choices = ['en', 'es'],
                        default = 'en',
                        help="Language of data (en or es)")

    parser.add_argument('-bs', '--batch-size',
                        type = int,
                        default = 16)
    
    args = vars(parser.parse_args())
    
    if args['config'] is not None:
        with open(args['config'], 'r') as f:
            args.update(json.load(f))
    
    return args

def load_data(args):
    base = args['data_dir']
    
    data = pd.read_csv(os.path.join(base, FP_TEMPLATE.format(language = args['language'])))

    return data

class PromptDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]
        
def collate_fn_instruct(batch_texts, tokenizer):
    chat_prompt = tokenizer.apply_chat_template(
                                batch_texts,
                                add_generation_prompt = True,
                                tokenize = False,
                            )

    tok_batch = tokenizer(
        chat_prompt,
        return_tensors = 'pt',
        padding = True
    )
                
    return tok_batch

def collate_fn_base(batch_texts, tokenizer):
    tok_batch = tokenizer(
        batch_texts,
        return_tensors="pt",
        padding=True,
    )

    return tok_batch

def load_model(args):
        
    model_id = args['model']
        
    if "chat" in model_id.lower():
        tokenizer = LlamaTokenizer.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf", 
            load_from_cache=False,
            keep_in_memory=True,
            padding_side = 'left',
        )
        model = LlamaForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16,
            device_map='auto'
        )

        tokenizer.pad_token = tokenizer.eos_token
        
    elif "mt5" in model_id.lower():
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, 
            model_max_length=512
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id, 
            use_cache=False, 
            device_map='auto'
        )
    elif "ministral-3" in model_id.lower():
        tokenizer = MistralCommonBackend.from_pretrained(model_id, 
             padding_side = 'left')

        model = Mistral3ForConditionalGeneration.from_pretrained(model_id, device_map = 'auto')
        
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, 
            keep_in_memory=True,
            padding_side = 'left',
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    return tokenizer, model, terminators

def make_prompt(args, aff_state, user_prompt, model):
    model_name = model.config.name_or_path.lower().split('/')[-1]

    instruction   = INSTRUCTIONS[args['language']].format(affective_state = aff_state)
    aff_state = aff_state[0].upper() + aff_state[1:]
          
    if ("instruct" in model_name) or ("chat" in model_name):

        messages = [
            {"role": "user", "content": f'{instruction}\nUser Message: "{user_prompt}"'}
        ]
        prompt_text = messages
    else:

        prompt_text = (
            f"{instruction}\n"
            f"User Message: \"{user_prompt}\"\n{aff_state} Response: \""
        )
    
    return prompt_text

def get_dataloader(args, data, model, tokenizer):
    model_name = model.config.name_or_path.lower().split('/')[-1]

    if ("instruct" in model_name) or ("chat" in model_name):
        def collate_fn(d):
            return collate_fn_instruct(d, tokenizer)
    else:
        def collate_fn(d):
            return collate_fn_base(d, tokenizer)

    dataset = PromptDataset(data)
    
    dataloader = DataLoader(
        data,
        batch_size=args['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last = False
    )

    return dataloader

def get_model_response(model, tokenizer, input_ids):
    model_name = model.config.name_or_path.lower().split('/')[-1]

    if not isinstance(input_ids, dict):
        input_ids = {'input_ids': input_ids}

    with torch.no_grad():
        outputs = model.generate(**input_ids,
                            max_new_tokens = 128,
                            do_sample = False)

    if 't5' not in model_name:
        outputs = outputs[:,input_ids['input_ids'].shape[1]:]
    response = tokenizer.batch_decode(outputs)
        
    return response

def get_model_response_row(args, row, model, tokenizer):
    aff_state = row['affective_state']
    prompt = row['message']

    prompt_text, tok_prompt = make_tok_prompt(args, aff_state, prompt, model, tokenizer)
    response = get_model_response(model, tokenizer, tok_prompt)
    
    row['prompt_text'] = prompt_text
    row['response'] = response

    return row

def get_res_path(args, model):
    model_name = model.config.name_or_path.lower().split('/')[-1]
        
    res_path = os.path.join(args['results_dir'], args['run_id'], model_name, 'expression')
    os.makedirs(res_path, exist_ok = True)
    
    return res_path

def main():
    args = parse_args()
    print('Parsed arguments: \n' + "\n".join(['\t' + str(name) + ': ' + str(val) for name, val in args.items()]))

    data = load_data(args)
    print('Data Loaded')

    tokenizer, model, terminators = load_model(args)
    model_name = model.config.name_or_path.lower().split('/')[-1]
    print('Loaded model and tokenizer')

    data['prompt'] = data.progress_apply(lambda row: make_prompt(args, row['affective_state'], row['message'], model), axis = 1)
    dataloader = get_dataloader(args, data['prompt'], model, tokenizer)
    print('Created Data Loader')
    
    full_responses, all_responses = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = {k: v.to(model.device, non_blocking=True) for k, v in batch.items()}
            
            out = model.generate(
                input_ids = batch['input_ids'],
                attention_mask = batch['attention_mask'],
                max_new_tokens=128,
                do_sample=True,
                temperature = 0.7,
                top_p = 0.9,
                no_repeat_ngram_size=4,
            )

            prompt_lens = [batch['input_ids'].shape[1]] * batch['input_ids'].shape[0]
            full_out_text = tokenizer.batch_decode(out)

            out = [
                        output[prompt_len:]
                        for output, prompt_len in zip(out, prompt_lens)
                    ]
            
            out_text = tokenizer.batch_decode(
                out,
                skip_special_tokens=True
            )

            full_responses += full_out_text
            all_responses += out_text

            del batch
            del out
            torch.cuda.empty_cache()

    data['full_response'] = full_responses
    data['response'] = all_responses
    
    res_path = os.path.join(get_res_path(args, model), f'{args["language"]}_preds.csv')
    data.to_csv(res_path, index = None)

    print('Done')
    
if __name__ == '__main__':
    main()