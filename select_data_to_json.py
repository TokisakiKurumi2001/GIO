import json
import torch

DATA_PATH='data/'

if __name__ == "__main__":
    indicies = torch.load(DATA_PATH + 'select.pt').numpy()
    
    data = []
    with open(DATA_PATH + 'syn.jsonl') as fin:
        for line in fin:
            d = json.loads(line)
            data.append(d)

    select_samples = []
    for idx in indicies:
        select_samples.append(data[idx])

    with open(DATA_PATH + 'gio_final.jsonl', 'w+') as fout:
        for d in select_samples:
            fout.write(json.dumps(d) + "\n")