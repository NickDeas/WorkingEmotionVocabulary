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
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaTokenizer, LlamaForCausalLM
# from transformers import MT5TokenizerFast, MT5ForConditionalGeneration
from transformers.models.mt5.modeling_mt5 import MT5ForConditionalGeneration
from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend


from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

import json
from datetime import datetime

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
np.random.seed(SEED)

SYSTEM_PROMPTS = {
    'en': 'You are an expert in human emotions and feelings.\n',
    'es': 'You are a bilingual expert in human emotions, feelings, and Spanish.\n'
}

INSTRUCTIONS_YN = (
    "Affective state refers to  any terms that humans use to describe their experiences of feeling, including emotions, moods, and figurative expressions of feelings (e.g. 'blue' as an expression of sadness instead of the color)."
    "Tell me if the word between <span> and </span> is an affective state or not. "
    "Output only Y if it is an affective state or N if it is not.\n"
    "Do not explain or preface your answer.\n"
    )

INSTRUCTIONS_SCORE = (
    "Affective state refers to  any terms that humans use to describe their experiences of feeling, including emotions, moods, and figurative expressions of feelings (e.g. 'blue' as an expression of sadness instead of the color)."
    "Does the term between <span> and </span> reflect an affective state? "
    "Answer with only one character from the following: 0, 1, 2, or 3.\n"
    "Where:\n"
    "0 means Not an affective state: the term does not refer to an emotion, feeling, or internal state.\n"
    "1 means Unlike an affective state: the term is referencing something that is not an emotion.\n"
    "2 means Like an affective state: the term is likely referencing an emotion, feeling, or internal state.\n"
    "3 means Completely an affective state: the term is definitely an emotion, feeling, or internal state.\n"
    "Do not explain or preface your answer.\n"

)

RESPONSES = {
    'yn': ['Y', 'N'],
    'score': [f'{d}' for d in range(4)]
}

FILENAMES = {
    'eval': '{language}_full.csv',
    'pool': '{language}_non_overlapping_annotations.csv'
}

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run affective state annotation experiment."
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
    
    parser.add_argument('-pt', '--prompt-types',
                        type = str,
                        nargs = '+',
                        default = ['yn', 'score'],
                        choices = ['yn', 'score'],
                        help = 'Task types (either yes/no or score)')
    
    parser.add_argument('-ks', "--ks", 
                        type=int, 
                        nargs = '+',
                        default=[0],
                        help="Numbers of few-shot examples (default: 0)")
    
    parser.add_argument('-l', "--language", 
                        type=str, 
                        choices = ['en', 'es'],
                        default = 'en',
                        help="Language of data (en or es)")
    
    args = vars(parser.parse_args())
    
    if args['config'] is not None:
        with open(args['config'], 'r') as f:
            args.update(json.load(f))
    
    return args


def presample_few_shot_indices(judgments_df, pool_df, num_samples=5):
    all_indices = pool_df.index.tolist()
    return [random.sample(all_indices, num_samples) for _ in range(len(judgments_df))]

def clean_html_preserve_spans(html_text):

    soup = BeautifulSoup(html_text, "html.parser")
    
    for tag in soup.find_all(True):
        if tag.name == "span":
            tag.attrs = {}
        else:
            tag.unwrap()
    
    text = str(soup)
    
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    
    return text

def load_data(args):
    base = args['data_dir']
    
    judgments_data = FILENAMES['eval'].format(language = args['language'])
    non_overlapping_data = FILENAMES['pool'].format(language = args['language'])

    judgments_df = pd.read_csv(os.path.join(base, judgments_data))
    non_overlapping_annotations_df = pd.read_csv(os.path.join(base, non_overlapping_data))
    
    judgments_df["clean_short_context"] = judgments_df["short_context"].apply(clean_html_preserve_spans)       
    
    non_overlapping_annotations_df["clean_short_context"] = non_overlapping_annotations_df["short_context"].apply(clean_html_preserve_spans)
    non_overlapping_annotations_df["annotator_YN"] = non_overlapping_annotations_df["affect_score"].apply(binarize_affect_score)

    judgments_df["few_shot_ids"] = presample_few_shot_indices(
        judgments_df, non_overlapping_annotations_df, num_samples=max(args['ks'])
    )
    
    return judgments_df, non_overlapping_annotations_df

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
    
    return tokenizer, model

def make_tok_prompt(args, model, tokenizer, examples, user_prompt, ptype = 'yn'):
    
    model_name = model.config.name_or_path.lower().split('/')[-1]
    language = args['language']
    
    system_prompt = SYSTEM_PROMPTS[language]
        
    instructions = INSTRUCTIONS_YN if ptype == 'short' else INSTRUCTIONS_SCORE
        
    if 'mt5' in model_name:
        prompt_text = (
            f"{instructions}\n"
            f"{examples}"
            f"Input text: {user_prompt}\nResponse:"
        )
        
        input_ids = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        
    elif ("instruct" in model_name) or ("chat" in model_name):

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": (
                f"{instructions}\n"
                f"{examples}\n"
                f"Input text: {user_prompt}\nResponse:\n"
            )}
        ]

        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)

        prompt_text = messages
    else:

        prompt_text = (
            f"System: {system_prompt}\n"
            f"User: {instructions}\n"
            f"{examples}\n"
            f"Input text: {user_prompt}\nResponse:\n"
        )

        input_ids = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        
    return prompt_text, input_ids

def log_prompt_settings(name, prompt_text):
    if isinstance(prompt_text, str):
        content = prompt_text
    else:
        content = [{"role": m["role"], "content": m["content"]} for m in prompt_text]

    experiment_settings[name] = {
        "input_to_model": content
    }

def get_res_path(args, model):
    model_name = model.config.name_or_path.lower().split('/')[-1]
        
    res_path = os.path.join(args['results_dir'], args['run_id'], model_name)
    os.makedirs(res_path, exist_ok = True)
    for k in args['ks']:
        os.makedirs(os.path.join(res_path, f'{k}_shot'), exist_ok = True)
    
    return res_path
        
def binarize_affect_score(score):
    """
    Convert an integer 0–3 rating into "Y"/"N".
    Example rule: score >=1.5 => "Y", else "N".
    """
    return "Y" if score >= 1.5 else "N"


### EXPERIMENTS ###

def get_few_shot_examples(df, ids, k, etype = 'yn'):
    example_ids = ids[:k]
    examples = ""
    for idx in example_ids:
        row = df.loc[idx]
        if etype == 'yn':
            examples += f"Input text: {row['clean_short_context']}\nResponse: {row['annotator_YN']}\n"
        else:
            examples += f"Input text: {row['clean_short_context']}\nResponse: {row['affect_score']}\n"
    
    return examples

def get_model_probs(model, input_ids):
    model_name = model.config.name_or_path.lower().split('/')[-1]

    if "t5" in model_name:

        with torch.no_grad():
            # Forward pass (mT5 is encoder-decoder, so logits start at decoder)
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=1,
                output_scores=True,
                return_dict_in_generate=True,
                do_sample=False  # deterministic
            )
    
        # Get logits for the first generated token
        logits = outputs.scores[0][0]  # shape: [vocab_size]

    if 'chat' in model_name or 'ministral-3' in model_name or 'mistral' in model_name:
        with torch.no_grad():
            outputs = model(**input_ids)

        logits = outputs.logits[0, -1, :]
    else:
        with torch.no_grad():
            outputs = model(**input_ids)
    
        logits = outputs.logits[0, -1, :]
        
    probs = F.softmax(logits, dim=-1)
    
    return probs

def get_prob_dict(tokenizer, probs, toks):
    
    tok_ids = {tok:tokenizer.convert_tokens_to_ids(tok) for tok in toks}
    prob_dict = {tok:probs[tok_id].item() for tok,tok_id in tok_ids.items()}
    
    return prob_dict

def get_model_pred(args, user_prompt, model, tokenizer, few_shot_df, few_shot_ids, k, ptype = 'yn'):
    model_name = model.config.name_or_path.lower().split('/')[-1]

    examples = get_few_shot_examples(few_shot_df, few_shot_ids, k, etype = ptype)
    prompt, tok_prompt = make_tok_prompt(args, model, tokenizer, examples, user_prompt, ptype = ptype)
    
    probs   = get_model_probs(model, tok_prompt)            
    prob_dict = get_prob_dict(tokenizer, probs, toks = RESPONSES[ptype])
    
    return prompt, prob_dict

def get_model_pred_row(args, row, user_prompt, model, tokenizer, few_shot_df, few_shot_ids, k, ptype = 'yn'):
    prompt_text, prob_dict = get_model_pred(args, user_prompt, model, tokenizer, few_shot_df, few_shot_ids, k, ptype = ptype)
    row['prompt'] = prompt_text
    row[f'{ptype}_probs'] = prob_dict
    return row


def normalize_yes_probability(prob_dict):
    probs = prob_dict['Y'], prob_dict['N']
    total = sum(probs) if sum(probs) != 0 else 0.5
    return probs[0]/total

def compute_weighted_score(prob_dict):
    """
    Given a dictionary like {"0": p0, "1": p1, "2": p2, "3": p3},
    normalize them, then compute: [(p0 * 1) + (p1 * 2) + (p2 * 3) + (p3 * 4)] / 4
    """
    
    values = [prob_dict.get(f'{d}', 0.0) for d in range(4)]
    total  = sum(values) if sum(values) != 0 else np.nan
    weighted_score = sum([d*v/total for d,v in zip(range(4), values)])/4
    
    return weighted_score

def get_prediction(prob_dict):
    y_val = prob_dict["Y"]
    n_val = prob_dict["N"]
    return "Y" if y_val >= n_val else "N"  # pick "Y" on tie

def binarize_affect_score(score):
    """
    Convert an integer 0–3 rating into "Y"/"N".
    Example rule: score >=1.5 => "Y", else "N".
    """
    return "Y" if score >= 1.5 else "N"

def extract_corr_and_p(val):
    """Parse 'correlation: 0.5123, p-value: 0.0234' into two floats"""
    match = re.search(r"correlation: ([\d\.\-e]+), p-value: ([\d\.\-e]+)", val)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None, None

def calc_corrs(preds, annot_a, annot_b, annot_avg):
    corr_res = {}
    
    res_a = spearmanr(preds, annot_a)
    corr_res[f'corr_a'], corr_res[f'corr_a_p'] = res_a.statistic, res_a.pvalue
    res_b = spearmanr(preds, annot_b)
    corr_res[f'corr_b'], corr_res[f'corr_b_p'] = res_b.statistic, res_b.pvalue
    res_avg = spearmanr(preds, annot_avg)
    corr_res[f'corr'], corr_res[f'corr_p'] = res_avg.statistic, res_avg.pvalue
    
    return corr_res

def calc_kappas(preds, annot_a, annot_b, annot_avg, binary = False):
    kappa_res = {}
    prefix = 'bin' if binary else ''
    
    kappa_res[f'{prefix}_kappa_a'] = cohen_kappa_score(preds, annot_a)
    kappa_res[f'{prefix}_kappa_b'] = cohen_kappa_score(preds, annot_b)
    kappa_res[f'{prefix}_kappa']   = cohen_kappa_score(preds, annot_avg)
    
    return kappa_res

def eval_yn(yn_preds, judgments_df):
    all_res = {}
    
    yn_probs = yn_preds.apply(normalize_yes_probability)
    yn_bin   = yn_preds.apply(lambda pdict: max(pdict, key=pdict.get))
    
    # Correlations
    prob_corr = calc_corrs(yn_probs, 
                         judgments_df["affect_score_a"],
                         judgments_df["affect_score_b"],
                         judgments_df["affect_score"])
    all_res['yn_prob'] = prob_corr
    
    bin_corr  = calc_corrs(yn_bin,
                         judgments_df["affect_score_a"],
                         judgments_df["affect_score_b"],
                         judgments_df["affect_score"])
    all_res['yn_class'] = bin_corr
        
    # Agreement
    kappa_bin = calc_kappas(yn_bin,
                         judgments_df["affect_bin_a"],
                         judgments_df["affect_bin_b"],
                         judgments_df["affect_bin"])
    all_res['yn_class'].update(kappa_bin)
    
    return all_res
    
        
def eval_score(score_preds, judgments_df):
    all_res = {}
    
    score_max = score_preds.apply(lambda pdict: int(max(pdict, key=pdict.get)))
    score_bin = score_max.apply(binarize_affect_score)
    score_weighted = score_preds.apply(compute_weighted_score)
    
    # Correlations
    corr_prob = calc_corrs(score_weighted, 
                         judgments_df["affect_score_a"],
                         judgments_df["affect_score_b"],
                         judgments_df["affect_score"])
    all_res['score_prob'] = corr_prob
    
    corr_max  = calc_corrs(score_max,
                         judgments_df["affect_score_a"],
                         judgments_df["affect_score_b"],
                         judgments_df["affect_score"])
    all_res['score_class'] = corr_max
        
    # Agreement
    kappa = calc_kappas(score_max,
                         judgments_df["affect_score_a"].round().astype(int),
                         judgments_df["affect_score_b"].round().astype(int),
                         judgments_df["affect_score"].round().astype(int))
    all_res['score_class'].update(kappa)
    
    kappa_bin = calc_kappas(score_bin,
                         judgments_df["affect_bin_a"],
                         judgments_df["affect_bin_b"],
                         judgments_df["affect_bin"],
                         binary = True)
    all_res['score_class'].update(kappa_bin)
    
    return all_res

def main():
    args = parse_args()
    print('Parsed arguments: \n' + "\n".join(['\t' + str(name) + ': ' + str(val) for name, val in args.items()]))
    
    judgments_df, non_overlapping_annotations_df = load_data(args)
    judgments_df["affect_bin_a"] = judgments_df["affect_score_a"].apply(binarize_affect_score)
    judgments_df["affect_bin_b"] = judgments_df["affect_score_b"].apply(binarize_affect_score)
    judgments_df["affect_bin"] = judgments_df["affect_score"].apply(binarize_affect_score)
    print('Data Loaded')
    
    tokenizer, model = load_model(args)
    print('Loaded model and tokenizer') 
    
    res_path = get_res_path(args, model)
    
    for k in args['ks']:
        print(f'Beginning {k}-shot Evaluation...')
        
        model_res = {}
        
        if 'yn' in args['prompt_types']:
            judgments_df_res = judgments_df.progress_apply(lambda row: get_model_pred_row(args, 
                                                                                row,
                                                                                row["clean_short_context"], 
                                                                                model, 
                                                                                tokenizer,
                                                                                non_overlapping_annotations_df, 
                                                                                row["few_shot_ids"], 
                                                                                k=k,
                                                                                ptype = 'yn'), 
                                              axis = 1)
            
            model_res.update(eval_yn(judgments_df_res['yn_probs'], judgments_df_res))
            
        if 'score' in args['prompt_types']:
            judgments_df_res = judgments_df.progress_apply(lambda row: get_model_pred_row(args, 
                                                                                row,
                                                                                row["clean_short_context"], 
                                                                                model, 
                                                                                tokenizer,
                                                                                non_overlapping_annotations_df, 
                                                                                row["few_shot_ids"], 
                                                                                k=k,
                                                                                ptype = 'score'), 
                                              axis = 1)
            
            model_res.update(eval_score(judgments_df_res['score_probs'], judgments_df_res))
        
        model_res = pd.DataFrame.from_dict(model_res, orient = 'index').reset_index()
        model_res = model_res.rename({'index': 'case'}, axis = 1)
        model_res.to_csv(os.path.join(res_path, f'{k}_shot', f'{args["language"]}_metrics.csv'), index = None)
        judgments_df_res.to_csv(os.path.join(res_path, f'{k}_shot', f'{args["language"]}_preds.csv'), index = None)

    print('Completed evaluation')
    
if __name__ == '__main__':
    main()
    
    