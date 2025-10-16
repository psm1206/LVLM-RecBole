import os
from utils.dataset import seed_everything, parse_args

from transformers import AutoModel
import json
import torch
from tqdm import tqdm
import pickle
import numpy as np


def generate_item_embedding_with_nvemb(item_metatext_dict, text_encoder, dataset, batch_size=5):
    passage_prefix = ''
    embeddings = {}
    start = 0
    text_list = list(item_metatext_dict.values())
    num_data = len(item_metatext_dict.values())
    id_list = list(item_metatext_dict.keys())

    with torch.autocast('cuda'):
        with tqdm(total=num_data, desc="Processing", unit="item") as pbar:
            while start < num_data:
                sentences = [text_list[start]]
                torch.cuda.empty_cache()
                ids = id_list[start]
                
                query_embeddings = text_encoder.encode(sentences, instruction=passage_prefix, max_length=32768)
                query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)
                tensor_embeddings = query_embeddings.detach().cpu().numpy()
                
                embeddings[ids] = tensor_embeddings

                del query_embeddings
                
                start += 1
                pbar.update(1)

                
    # embeddings = np.concatenate(embeddings, axis=0)

    id_map = json.load(open(f"./{dataset}/id_map.json", "r"))["id2item"]
    emb_list = []
    for id in range(1, len(embeddings)+1):
        meta_emb = embeddings[id_map[str(id)]][0]
        emb_list.append(meta_emb)

    emb_list = np.array(emb_list)

    root_path = os.getcwd()
    save_path = os.path.join(root_path, f'{dataset}')
    os.makedirs(save_path, exist_ok=True)

    with open(os.path.join(save_path, f'nvembv2.pkl'), 'wb') as f:
        pickle.dump(emb_list, f)

    del embeddings


if __name__ == '__main__':
    args = parse_args()
        
    dataset = args.dataset
    data_path = os.path.join('.', f'{dataset}/item_str.json')
    seed_everything(42)


    print("Load Text Encoder")
    model = AutoModel.from_pretrained('nvidia/NV-Embed-v2', trust_remote_code=True, device_map="auto", torch_dtype="auto")
    
    item_metatext_dict = json.load(open(data_path))
    generate_item_embedding_with_nvemb(item_metatext_dict, model, dataset, batch_size=2)
    