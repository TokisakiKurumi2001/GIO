from transformers import AutoTokenizer, AutoModelForCausalLM 
import json, os
import torch
from loguru import logger
from tqdm import tqdm
from typing import List

DATA_PATH='data/syn.jsonl'
SAVE_PATH='data/syn.pt'
MODEL_PATH='llama_2'
PEFT_PATH='teacher_sft'
MAX_LENGTH=1024

def encode(data, tokenizer, model, batch_size: int=4) -> List:
    device = model.device
    num_sample = len(data)
    preds = []
    for i in tqdm(range(0, num_sample, batch_size), desc='Encoding', unit='batch'):
        batch = data[i:i+batch_size]
        texts = [el['prompt'] for el in batch]
        inputs = tokenizer(texts, return_tensors='pt', padding=True, max_length=MAX_LENGTH, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = model(**inputs).last_hidden_state
        sent_len = torch.sum(inputs['attention_mask'], dim=1)
        batch_indicies = torch.arange(batch_size)
        pred = outputs[batch_indicies, sent_len].squeeze()
        preds.append(pred.detach().clone().cpu())

        del inputs
        del outputs
        del pred
        exit()

    pred = torch.stack(preds)
    return pred

if __name__ == "__main__":
    logger.info("Loading data ...")
    data = []
    with open(DATA_PATH) as fin:
        for line in fin:
            d = json.loads(line)
            data.append(d)

    logger.success(f'Successfully load {len(data)} samples.')

    logger.info('Loading model ...')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    model.load_adapter(PEFT_PATH)
    device = torch.device('cuda:0')
    model = model.to(device)
    model.eval()
    logger.success(f'Successfully load {MODEL_PATH} model.')

    logger.info('Encoding ...')
    pred = encode(data, tokenizer, model)
    torch.save(pred, SAVE_PATH)
