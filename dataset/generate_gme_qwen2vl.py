import os
import json
import ast
import pickle
import gc
import pandas as pd
from typing import Dict, List, Tuple

import requests
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel
from transformers.utils.versions import require_version
 

from utils.dataset import parse_args, seed_everything, amazon_dataset2fullname
from utils.text import clean_metadata


require_version(
    "transformers<4.52.0",
    "The remote code has some issues with transformers>=4.52.0, please downgrade: pip install transformers==4.51.3"
)


# Disable tokenizers parallelism warning
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

# Patch requests.get globally to ensure proper headers/timeout for image URLs
_orig_requests_get = requests.get

def _patched_requests_get(url, *args, **kwargs):
    headers = kwargs.pop('headers', None) or {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36'
    }
    timeout = kwargs.pop('timeout', 30)
    resp = _orig_requests_get(url, *args, headers=headers, timeout=timeout, stream=kwargs.get('stream'))
    return resp

requests.get = _patched_requests_get


def _flatten_categories(categories_value) -> str:
    if isinstance(categories_value, str):
        return categories_value
    if isinstance(categories_value, list):
        parts: List[str] = []
        for item in categories_value:
            if isinstance(item, str):
                if item.strip():
                    parts.append(item.strip())
            elif isinstance(item, list):
                parts.extend([str(x).strip() for x in item if str(x).strip()])
            else:
                s = str(item).strip()
                if s:
                    parts.append(s)
        return ', '.join(parts)
    if isinstance(categories_value, dict):
        parts = [str(v).strip() for v in categories_value.values() if str(v).strip()]
        return ', '.join(parts)
    return ''


def _build_text(obj: dict, dataset_full_name: str) -> str:
    title = str(obj.get('title', '') or '').strip()
    price = str(obj.get('price', '') or '').strip()
    brand = str(obj.get('brand', '') or '').strip()
    categories = _flatten_categories(obj.get('categories', ''))
    description = str(obj.get('description', '') or '').strip()

    segments: List[str] = []
    if title:
        segments.append(f"title is {title}")
    if price:
        segments.append(f"price is {price}")
    if brand:
        segments.append(f"brand is {brand}")
    if categories:
        segments.append(f"categories is {categories}")
    if description:
        segments.append(f"description is {description}")

    if not segments:
        return ''
    return f"The {dataset_full_name} item has the following attributes: \n " + '; '.join(segments)


def load_metadata(dataset_name: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    dataset_full_name = amazon_dataset2fullname[dataset_name]
    meta_path = os.path.join(dataset_name, f'meta_{dataset_full_name}.json')
    asin_to_text = {}
    asin_to_image = {}

    # 1) 라인 파싱 후 DataFrame 구성
    rows: List[Dict[str, object]] = []
    with open(meta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                try:
                    obj = ast.literal_eval(line)
                except (ValueError, SyntaxError):
                    continue

            asin = obj.get('asin') or obj.get('ASIN')
            if not asin:
                continue

            rows.append({
                'asin': asin,
                'title': obj.get('title', ''),
                'price': obj.get('price', ''),
                'brand': obj.get('brand', ''),
                'categories': obj.get('categories', ''),
                'description': obj.get('description', ''),
                'imUrl': obj.get('imUrl', ''),
            })

    if not rows:
        return asin_to_text, asin_to_image

    df = pd.DataFrame(rows)

    string_cols = ['asin', 'title', 'price', 'brand', 'categories', 'description', 'imUrl']
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).apply(lambda x: x.strip())

    # 2) 텍스트 전처리
    df = clean_metadata(df)

    # 3) 포맷팅 및 사전 구축
    # categories 문자열에 남아있는 선두/말미의 [[, ]] 제거
    if 'categories' in df.columns:
        df['categories'] = df['categories'].astype(str)
        df['categories'] = df['categories'].str.replace(r'^\[\[', '', regex=True)
        df['categories'] = df['categories'].str.replace(r'\]\]$', '', regex=True)

    for _, row in df.iterrows():
        asin = row['asin']
        image_url = row.get('imUrl', '')
        if isinstance(image_url, str) and image_url:
            asin_to_image[asin] = image_url

        obj_clean = {
            'title': row.get('title', ''),
            'price': row.get('price', ''),
            'brand': row.get('brand', ''),
            'categories': row.get('categories', ''),
            'description': row.get('description', ''),
        }
        text_formatted = _build_text(obj_clean, dataset_full_name)
        if text_formatted:
            asin_to_text[asin] = text_formatted

    return asin_to_text, asin_to_image


def load_id2item(dataset_name: str) -> Dict[str, str]:
    id_map_path = os.path.join(dataset_name, 'id_map.json')
    with open(id_map_path, 'r') as f:
        obj = json.load(f)
    return obj['id2item']


def batched(iterable: List, batch_size: int) -> List[List]:
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]


def generate_embeddings_with_gme(model, asin_to_text, asin_to_image, batch_size=8, modality='text'):
    embeds = {}
    error_count = 0

    if modality == 'text':
        asins_with_text = [asin for asin, txt in asin_to_text.items() if isinstance(txt, str) and txt.strip()]
        for batch_asins in tqdm(list(batched(asins_with_text, batch_size)), desc='Text embeddings'):
            texts = [asin_to_text[a] for a in batch_asins]
            with torch.inference_mode():
                emb = model.get_text_embeddings(texts=texts)
            emb_np = emb.detach().cpu().numpy()
            for a, e in zip(batch_asins, emb_np):
                embeds[a] = e

    elif modality == 'image':
        asins_with_image = [asin for asin, url in asin_to_image.items() if isinstance(url, str) and url.strip()]
        for batch_asins in tqdm(list(batched(asins_with_image, batch_size)), desc='Image embeddings'):
            urls = [asin_to_image[a] for a in batch_asins]
            try:
                with torch.inference_mode():
                    emb = model.get_image_embeddings(images=urls)
                emb_np = emb.detach().cpu().numpy()
                for a, e in zip(batch_asins, emb_np):
                    embeds[a] = e
            except Exception:
                # Fallback to per-item to salvage valid ones quickly
                for a in batch_asins:
                    try:
                        with torch.inference_mode():
                            emb = model.get_image_embeddings(images=[asin_to_image[a]])
                        embeds[a] = emb.detach().cpu().numpy()[0]
                    except Exception:
                        error_count += 1
                        print(f"Error getting image embeddings for {a}")
                        continue

    elif modality == 'fused':
        asins_with_text = [asin for asin, txt in asin_to_text.items() if isinstance(txt, str) and txt.strip()]
        asins_with_both = [a for a in asins_with_text if a in asin_to_image]
        for batch_asins in tqdm(list(batched(asins_with_both, batch_size)), desc='Fused embeddings'):
            texts = [asin_to_text[a] for a in batch_asins]
            urls = [asin_to_image[a] for a in batch_asins]
            try:
                with torch.inference_mode():
                    emb = model.get_fused_embeddings(texts=texts, images=urls)
                emb_np = emb.detach().cpu().numpy()
                for a, e in zip(batch_asins, emb_np):
                    embeds[a] = e
            except Exception:
                # Fallback per-item
                for a in batch_asins:
                    try:
                        with torch.inference_mode():
                            emb = model.get_fused_embeddings(texts=[asin_to_text[a]], images=[asin_to_image[a]])
                        embeds[a] = emb.detach().cpu().numpy()[0]
                    except Exception:
                        error_count += 1
                        print(f"Error getting fused embeddings for {a}")
                        continue
    else:
        raise ValueError(f"Invalid modality: {modality}")
    
    print(f"{modality} embeddings error count: {error_count}")
    print(f"{modality} embeddings error rate: {error_count / len(embeds) * 100:.2f}%")

    return embeds


def to_ordered_array(embeds, id2item, fill_dim: int) -> np.ndarray:
    size = len(id2item)
    arr = np.zeros((size, fill_dim), dtype=np.float16)
    for i in range(1, size + 1):
        asin = id2item[str(i)]
        if asin in embeds:
            vec = embeds[asin]
            arr[i - 1] = vec.astype(np.float16)
    return arr


def main():
    seed_everything(42)
    args = parse_args()
    dataset_name = args.dataset  # e.g., 'beauty'
    embedding_model = args.embedding_model
    batch_size = args.batch_size
    gpu_id = args.gpu_id

    short_name_dict = {'Alibaba-NLP/gme-Qwen2-VL-2B-Instruct': 'gme_qwen2vl2b',
                        'Alibaba-NLP/gme-Qwen2-VL-7B-Instruct': 'gme_qwen2vl7b'}

    embedding_model_saved_name = short_name_dict[embedding_model]

    asin_to_text, asin_to_image = load_metadata(dataset_name)
    id2item = load_id2item(dataset_name)

    # filtering asin_to_text and asin_to_image with id2item
    asin_to_text = {asin: text for asin, text in asin_to_text.items() if asin in id2item.values()}
    asin_to_image = {asin: url for asin, url in asin_to_image.items() if asin in id2item.values()}

    print(f'Load {embedding_model} model')
    device = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'
    model = AutoModel.from_pretrained(
        embedding_model,
        torch_dtype='float16',
        trust_remote_code=True,
    )
    model = model.to(device)

    # Determine dims using any available vector per modality
    def pick_dim(d) -> int:
        for v in d.values():
            return int(v.shape[-1])
        return 0

    save_dir = dataset_name
    os.makedirs(save_dir, exist_ok=True)

    # text embeddings
    text_emb_path = os.path.join(save_dir, f'{embedding_model_saved_name}_text.pkl')
    if os.path.exists(text_emb_path):
        print(f"Text embeddings already exist: {text_emb_path}")
    else:
        text_embeds = generate_embeddings_with_gme(
            model, asin_to_text, asin_to_image, batch_size=batch_size, modality='text'
        )
        text_dim = pick_dim(text_embeds)

        if text_dim > 0:
            text_arr = to_ordered_array(text_embeds, id2item, text_dim)
            with open(text_emb_path, 'wb') as f:
                pickle.dump(text_arr, f)
            print(f"Saved text embeddings: {text_arr.shape} -> {text_emb_path}")
        else:
            print('No text embeddings produced.')

    # image embeddings
    image_emb_path = os.path.join(save_dir, f'{embedding_model_saved_name}_image.pkl')
    if os.path.exists(image_emb_path):
        print(f"Image embeddings already exist: {image_emb_path}")
    else:
        image_embeds = generate_embeddings_with_gme(
            model, asin_to_text, asin_to_image, batch_size=batch_size, modality='image'
        )
        image_dim = pick_dim(image_embeds)

        if image_dim > 0:
            image_arr = to_ordered_array(image_embeds, id2item, image_dim)
            with open(image_emb_path, 'wb') as f:
                pickle.dump(image_arr, f)
            print(f"Saved image embeddings: {image_arr.shape} -> {image_emb_path}")
        else:
            print('No image embeddings produced.')

    # plot similarity between text and image embeddings
    from sklearn.metrics.pairwise import cosine_similarity
    import matplotlib.pyplot as plt

    # text_arr.shape: (item_num, dim)
    # image_arr.shape: (item_num, dim)
    text_arr = pickle.load(open(text_emb_path, 'rb'))
    image_arr = pickle.load(open(image_emb_path, 'rb'))
    similarity = cosine_similarity(text_arr, image_arr)
    plt.figure(figsize=(10, 10))
    plt.imshow(similarity, cmap='viridis')
    plt.colorbar()
    plt.savefig(os.path.join(save_dir, f'text_image_similarity_{dataset_name}_{embedding_model_saved_name}.png'))
    plt.close()

    del text_arr, image_arr, similarity
    gc.collect()
    torch.cuda.empty_cache()

    # fused embeddings
    fused_emb_path = os.path.join(save_dir, f'{embedding_model_saved_name}_fused.pkl')
    if os.path.exists(fused_emb_path):
        print(f"Fused embeddings already exist: {fused_emb_path}")
    else:
        fused_embeds = generate_embeddings_with_gme(
            model, asin_to_text, asin_to_image, batch_size=batch_size, modality='fused'
        )
        fused_dim = pick_dim(fused_embeds)

        if fused_dim > 0:
            fused_arr = to_ordered_array(fused_embeds, id2item, fused_dim)
            with open(fused_emb_path, 'wb') as f:
                pickle.dump(fused_arr, f)
            print(f"Saved fused embeddings: {fused_arr.shape} -> {fused_emb_path}")
        else:
            print('No fused embeddings produced.')


if __name__ == '__main__':
    main()
