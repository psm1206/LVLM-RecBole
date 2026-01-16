import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import ast
import pickle
import time
import pandas as pd
from typing import Dict, List, Tuple

import requests
import numpy as np
import torch
from tqdm import tqdm
from src.models.qwen3_vl_embedding import Qwen3VLEmbedder

from utils.dataset import parse_args, seed_everything, amazon_dataset2fullname
from utils.text import clean_metadata

"""
    transformers 4.51.0 -> Qwen2-VL
    transformers 4.57.0 -> Qwen3-VL
"""

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
    asin_to_title = {}
    asin_to_description = {}
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
        asin_to_text[asin] = text_formatted
        asin_to_title[asin] = row.get('title', '')
        asin_to_description[asin] = row.get('description', '')

    return asin_to_text, asin_to_title, asin_to_description, asin_to_image


def load_id2item(dataset_name: str) -> Dict[str, str]:
    id_map_path = os.path.join(dataset_name, 'id_map.json')
    with open(id_map_path, 'r') as f:
        obj = json.load(f)
    return obj['id2item']


def batched(iterable: List, batch_size: int) -> List[List]:
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]


def to_ordered_array(embeds, id2item, fill_dim: int) -> np.ndarray:
    size = len(id2item)
    arr = np.zeros((size, fill_dim), dtype=np.float16)
    for i in range(1, size + 1):
        asin = id2item[str(i)]
        if asin in embeds:
            vec = embeds[asin]
            arr[i - 1] = vec.astype(np.float16)
    return arr


# Determine dims using any available vector per modality
def pick_dim(d) -> int:
    for v in d.values():
        return int(v.shape[-1])
    return 0


def generate_embeddings_with_qwen3(model, asin_to_text1, asin_to_text2, asin_to_image, batch_size, modality, desc):
    embeds = {}
    error_count = 0
    
    # 1. 대상 ASIN 리스트 및 입력 데이터(List of Dict) 생성
    asins = []
    input_data = []

    if modality == 'text':
        asins = [a for a, t in asin_to_text1.items() if isinstance(t, str) and t.strip()]
        input_data = [{"text": asin_to_text1[a]} for a in asins]

    elif modality == 'image':
        asins = [a for a, u in asin_to_image.items() if isinstance(u, str) and u.strip()]
        input_data = [{"image": asin_to_image[a]} for a in asins]

    elif modality == 'text_with_text':
        asins = [a for a in asin_to_text1 if a in asin_to_text2 and asin_to_text1[a].strip()]
        input_data = [{"text": f"{asin_to_text1[a]} {asin_to_text2[a]}"} for a in asins]

    elif modality == 'text_with_image':
        asins = [a for a in asin_to_text1 if a in asin_to_image and asin_to_text1[a].strip()]
        input_data = [{"text": asin_to_text1[a], "image": asin_to_image[a]} for a in asins]

    # 2. 배치 단위 처리
    for i in tqdm(range(0, len(asins), batch_size), desc=f'{desc} Qwen3-VL-Embedding'):
        batch_asins = asins[i : i + batch_size]
        batch_inputs = input_data[i : i + batch_size]
        
        try:
            with torch.inference_mode():
                batch_embeds = model.process(batch_inputs)
            
            if isinstance(batch_embeds, torch.Tensor):
                batch_embeds = batch_embeds.cpu().numpy()
                
            for a, e in zip(batch_asins, batch_embeds):
                embeds[a] = e
        except Exception as e:
            print(f"Error in batch: {e}")
            error_count += len(batch_asins)
            continue
    
    return embeds


def generate_and_save_embeddings(
    embedding_dir: str,
    model,
    asin_to_text1: Dict,
    asin_to_text2: Dict,
    asin_to_image: Dict,
    batch_size: int,
    modality: str,
    desc: str,
    id2item: Dict[str, str],
    filename: str
):
    """
    Generate embeddings and save them if they don't already exist.
    
    Args:
        embedding_dir: Directory to save embeddings
        model: Embedding model
        asin_to_text1: First text dictionary
        asin_to_text2: Second text dictionary (for text_with_text modality)
        asin_to_image: Image dictionary (for image/text_with_image modalities)
        batch_size: Batch size for embedding generation
        modality: Modality type ('text', 'text_with_text', 'image', 'text_with_image')
        desc: Description for progress bar
        id2item: Mapping from item ID to ASIN
        filename: Filename to save embeddings (without extension)
    """
    emb_path = os.path.join(embedding_dir, f'{filename}.pkl')
    if os.path.exists(emb_path):
        print(f"{desc} embeddings already exist: {emb_path}")
    else:
        embeds = generate_embeddings_with_qwen3(
            model, asin_to_text1, asin_to_text2, asin_to_image, batch_size, modality, desc
        )
        emb_dim = pick_dim(embeds)
        emb_arr = to_ordered_array(embeds, id2item, emb_dim)
        with open(emb_path, 'wb') as f:
            pickle.dump(emb_arr, f)
        print(f"Saved {desc} embeddings: {emb_arr.shape} -> {emb_path}")


def main():
    seed_everything(42)
    args = parse_args()
    dataset_name = args.dataset  # e.g., 'beauty'
    embedding_model = args.embedding_model
    batch_size = args.batch_size
    gpu_id = args.gpu_id
    device = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'

    short_name_dict = {'Qwen/Qwen3-VL-2B-Instruct': 'qwen3vl2b',
                        'Qwen/Qwen3-VL-4B-Instruct': 'qwen3vl4b',
                        'Qwen/Qwen3-VL-8B-Instruct': 'qwen3vl8b',
                        'Qwen/Qwen3-VL-Embedding-2B': 'qwen3vl_emb2b',
                        'Qwen/Qwen3-VL-Embedding-8B': 'qwen3vl_emb8b'}

    embedding_model_saved_name = short_name_dict[embedding_model]

    save_dir = dataset_name
    os.makedirs(save_dir, exist_ok=True)
    id2item = load_id2item(dataset_name)

    # Check and load/create each asin_to file individually
    asin_to_text_path = os.path.join(save_dir, 'asin_to_text.json')
    asin_to_title_path = os.path.join(save_dir, 'asin_to_title.json')
    asin_to_description_path = os.path.join(save_dir, 'asin_to_description.json')
    asin_to_image_path = os.path.join(save_dir, 'asin_to_image.json')
    
    # Initialize dictionaries
    asin_to_text = {}
    asin_to_title = {}
    asin_to_description = {}
    asin_to_image = {}
    
    # Check which files need to be created
    need_to_load = []
    if not os.path.exists(asin_to_text_path):
        need_to_load.append('text')
    else:
        asin_to_text = json.load(open(asin_to_text_path, 'r', encoding='utf-8'))
        print(f"Loaded existing asin_to_text: {len(asin_to_text)} -> {asin_to_text_path}")
    
    if not os.path.exists(asin_to_title_path):
        need_to_load.append('title')
    else:
        asin_to_title = json.load(open(asin_to_title_path, 'r', encoding='utf-8'))
        print(f"Loaded existing asin_to_title: {len(asin_to_title)} -> {asin_to_title_path}")
    
    if not os.path.exists(asin_to_description_path):
        need_to_load.append('description')
    else:
        asin_to_description = json.load(open(asin_to_description_path, 'r', encoding='utf-8'))
        print(f"Loaded existing asin_to_description: {len(asin_to_description)} -> {asin_to_description_path}")
    
    if not os.path.exists(asin_to_image_path):
        need_to_load.append('image')
    else:
        asin_to_image = json.load(open(asin_to_image_path, 'r', encoding='utf-8'))
        print(f"Loaded existing asin_to_image: {len(asin_to_image)} -> {asin_to_image_path}")
    
    # If any files are missing, load metadata and create them
    if need_to_load:
        print(f"Loading metadata to create missing files: {', '.join(need_to_load)}")
        st_time = time.time()
        loaded_text, loaded_title, loaded_description, loaded_image = load_metadata(dataset_name)
        
        # Save only missing files
        if 'text' in need_to_load:
            asin_to_text = {asin: text for asin, text in loaded_text.items() if asin in id2item.values()}
            with open(asin_to_text_path, 'w', encoding='utf-8') as f:
                json.dump(asin_to_text, f, ensure_ascii=False, indent=2)
            print(f"Saved asin_to_text: {len(asin_to_text)} -> {asin_to_text_path}")
        
        if 'title' in need_to_load:
            asin_to_title = {asin: title for asin, title in loaded_title.items() if asin in id2item.values()}
            with open(asin_to_title_path, 'w', encoding='utf-8') as f:
                json.dump(asin_to_title, f, ensure_ascii=False, indent=2)
            print(f"Saved asin_to_title: {len(asin_to_title)} -> {asin_to_title_path}")
        
        if 'description' in need_to_load:
            asin_to_description = {asin: description for asin, description in loaded_description.items() if asin in id2item.values()}
            with open(asin_to_description_path, 'w', encoding='utf-8') as f:
                json.dump(asin_to_description, f, ensure_ascii=False, indent=2)
            print(f"Saved asin_to_description: {len(asin_to_description)} -> {asin_to_description_path}")
        
        if 'image' in need_to_load:
            asin_to_image = {asin: url for asin, url in loaded_image.items() if asin in id2item.values()}
            with open(asin_to_image_path, 'w', encoding='utf-8') as f:
                json.dump(asin_to_image, f, ensure_ascii=False, indent=2)
            print(f"Saved asin_to_image: {len(asin_to_image)} -> {asin_to_image_path}")
        
        print(f"Load + Save time taken: {time.time() - st_time:.2f} seconds")

    # Load embedding model
    print(f'Load {embedding_model} model')
    embedder = Qwen3VLEmbedder(
        model_name_or_path=embedding_model,
        torch_dtype=torch.float16,
        # attn_implementation="flash_attention_2" # 지원 시 활성
    )
    
    embedder.model.to(device)

    embedding_dir = os.path.join(save_dir, f'{embedding_model_saved_name}')
    os.makedirs(embedding_dir, exist_ok=True)

    # Text embeddings
    generate_and_save_embeddings(
        embedding_dir, embedder, asin_to_text, {}, {}, batch_size,
        modality='text', desc='Text', id2item=id2item, filename='text'
    )

    # Image embeddings
    generate_and_save_embeddings(
        embedding_dir, embedder, {}, {}, asin_to_image, batch_size,
        modality='image', desc='Image', id2item=id2item, filename='image'
    )

    # Text+Image embeddings
    generate_and_save_embeddings(
        embedding_dir, embedder, asin_to_text, {}, asin_to_image, batch_size,
        modality='text_with_image', desc='Text+Image', id2item=id2item, filename='text_image'
    )

if __name__ == '__main__':
    main()
