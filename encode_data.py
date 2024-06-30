from transformers import AutoTokenizer, AutoModelForCausalLM 
import json, os
import torch
from loguru import logger
from tqdm import tqdm
from typing import List

DATA_PATH='data/syn.jsonl'
SAVE_PATH='data/syn.pt'
MODEL_PATH='meta-llama/Llama-2-7b-hf'
PEFT_PATH='teacher_sft'

def encode(data, tokenizer, model, batch_size: int=16) -> List:
    device = model.device
    num_sample = len(data)
    preds = []
    for i in tqdm(range(0, num_sample, batch_size), desc='Encoding', unit='batch'):
        batch = data[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model(**inputs).logits
        sent_len = torch.sum(inputs['attention_mask'], dim=1)
        pred = torch.index_select(input=outputs, dim=1, indicies=sent_len)
        preds.append(pred.clone().cpu())

        del inputs
        del outputs
        del pred

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
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    model.load_adapter(PEFT_PATH)
    device = torch.device('cuda:0')
    model = model.to(device)
    model.eval()
    logger.success(f'Successfully load {MODEL_PATH} model.')

    logger.info('Encoding ...')
    pred = encode(data, tokenizer, model)
    torch.save(pred, SAVE_PATH)