import os, torch, re
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import pickle as pkl


def generate_text_multihot(metadata_df, word2idx, feature_list):
    rows, cols, data = [], [], []

    for item_idx, item_meta in metadata_df.iterrows():
        word_exists = defaultdict(int)

        for col in feature_list:
            text = str(item_meta.get(col, ""))
                
            for chunk in text.split(', '):
                for word in chunk.split():
                    word = word.lower().strip()
                    word = re.sub(r'[^a-z0-9]', '', word)
                    if not word:
                        continue

                    word_id = word2idx.get(word, -1)
                    if word_id == -1:
                        print("Word not found in word2idx:", word)
                    word_exists[word_id] = 1

        for word_id in word_exists.keys():
            rows.append(item_idx)
            cols.append(word_id)
            data.append(1)

    return rows, cols, data


def generate_item_embedding_with_nvemb(item_metatext_dict, text_encoder, dataset, batch_size=5,
                                                features=["title", "category", "brand", "description"]):
    # query_prefix = f"Instruct: Identify semantically similar items that users would browse together based on item information." + \
    #                 "\nItem informaion: "
    passage_prefix = ''
    embeddings = []
    start = 0
    text_list = list(item_metatext_dict.values())

    num_data = len(item_metatext_dict.values())

    with torch.autocast('cuda'):
        with tqdm(total=num_data, desc="Processing", unit="item") as pbar:
            while start < num_data:
                sentences = text_list[start: start + batch_size]
                torch.cuda.empty_cache()

                query_embeddings = text_encoder.encode(sentences, instruction=passage_prefix, max_length=32768)
                query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)
                tensor_embeddings = query_embeddings.detach().cpu().numpy()
                
                embeddings.append(tensor_embeddings)

                del query_embeddings
                
                start += batch_size
                pbar.update(batch_size)
                
    embeddings = np.concatenate(embeddings, axis=0)

    root_path = os.getcwd()
    save_path = os.path.join(root_path, f'data/{dataset}')
    os.makedirs(save_path, exist_ok=True)

    with open(os.path.join(save_path, f'nvembv2_items.{"_".join(features)}.pkl'), 'wb') as f:
        pkl.dump(embeddings, f)

    del embeddings

