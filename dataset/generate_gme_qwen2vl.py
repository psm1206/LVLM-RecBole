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


def generate_embeddings_with_gme(model, asin_to_text1, asin_to_text2, asin_to_image, batch_size, modality, desc):
    embeds = {}
    error_count = 0

    if modality == 'text':
        asins_with_text = [asin for asin, txt in asin_to_text1.items() if isinstance(txt, str) and txt.strip()]
        for batch_asins in tqdm(list(batched(asins_with_text, batch_size)), desc=f'{desc} Text embeddings'):
            texts = [asin_to_text1[a] for a in batch_asins]
            with torch.inference_mode():
                emb = model.get_text_embeddings(texts=texts)
            emb_np = emb.detach().cpu().numpy()
            for a, e in zip(batch_asins, emb_np):
                embeds[a] = e

    elif modality == 'image':
        asins_with_image = [asin for asin, url in asin_to_image.items() if isinstance(url, str) and url.strip()]
        for batch_asins in tqdm(list(batched(asins_with_image, batch_size)), desc=f'{desc} Image embeddings'):
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

    elif modality == 'text_with_text':
        asins_with_text1 = [asin for asin, txt in asin_to_text1.items() if isinstance(txt, str) and txt.strip()]
        asins_with_text2 = [asin for asin, txt in asin_to_text2.items() if isinstance(txt, str) and txt.strip()]
        asins_with_both = [a for a in asins_with_text2 if a in asins_with_text1]
        for batch_asins in tqdm(list(batched(asins_with_both, batch_size)), desc=f'{desc} Text+Text embeddings'):
            # concatenate text and text
            texts = [asin_to_text1[a] + ' ' + asin_to_text2[a] for a in batch_asins]
            with torch.inference_mode():
                emb = model.get_text_embeddings(texts=texts)
            emb_np = emb.detach().cpu().numpy()
            for a, e in zip(batch_asins, emb_np):
                embeds[a] = e                

    elif modality == 'text_with_image':
        asins_with_text = [asin for asin, txt in asin_to_text1.items() if isinstance(txt, str) and txt.strip()]
        asins_with_both = [a for a in asins_with_text if a in asin_to_image]
        for batch_asins in tqdm(list(batched(asins_with_both, batch_size)), desc=f'{desc} Text+Image embeddings'):
            texts = [asin_to_text1[a] for a in batch_asins]
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
                            emb = model.get_fused_embeddings(texts=[asin_to_text1[a]], images=[asin_to_image[a]])
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


def pick_dim(d) -> int:
    for v in d.values():
        return int(v.shape[-1])
    return 0


def generate_and_save_embeddings(
    embedding_dir: str,
    model,
    asin_to_text: Dict,
    asin_to_image: Dict,
    batch_size: int,
    modality: str,
    desc: str,
    id2item: Dict[str, str],
    filename: str
):
    emb_path = os.path.join(embedding_dir, f'{filename}.pkl')
    if os.path.exists(emb_path):
        print(f"{desc} embeddings already exist: {emb_path}")
        return

    embeds = generate_embeddings_with_gme(
        model, asin_to_text, asin_to_image, batch_size=batch_size, modality=modality
    )
    emb_dim = pick_dim(embeds)
    if emb_dim == 0:
        print(f"No {desc.lower()} embeddings produced.")
        return

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

    short_name_dict = {'Alibaba-NLP/gme-Qwen2-VL-2B-Instruct': 'gme_qwen2vl2b',
                        'Alibaba-NLP/gme-Qwen2-VL-7B-Instruct': 'gme_qwen2vl7b'}

    embedding_model_saved_name = short_name_dict[embedding_model]

    id2item = load_id2item(dataset_name)

    # 개별 파일 존재 여부 확인 후 로드/생성
    save_dir = dataset_name
    os.makedirs(save_dir, exist_ok=True)
    asin_to_text_path = os.path.join(save_dir, 'asin_to_text.json')
    asin_to_image_path = os.path.join(save_dir, 'asin_to_image.json')

    asin_to_text: Dict[str, str] = {}
    asin_to_image: Dict[str, str] = {}
    need_to_load: List[str] = []

    if os.path.exists(asin_to_text_path):
        asin_to_text = json.load(open(asin_to_text_path, 'r', encoding='utf-8'))
        print(f"Loaded existing asin_to_text: {len(asin_to_text)} -> {asin_to_text_path}")
    else:
        need_to_load.append('text')

    if os.path.exists(asin_to_image_path):
        asin_to_image = json.load(open(asin_to_image_path, 'r', encoding='utf-8'))
        print(f"Loaded existing asin_to_image: {len(asin_to_image)} -> {asin_to_image_path}")
    else:
        need_to_load.append('image')

    if need_to_load:
        print(f"Loading metadata to create missing files: {', '.join(need_to_load)}")
        loaded_text, loaded_image = load_metadata(dataset_name)

        if 'text' in need_to_load:
            asin_to_text = {asin: text for asin, text in loaded_text.items() if asin in id2item.values()}
            with open(asin_to_text_path, 'w', encoding='utf-8') as f:
                json.dump(asin_to_text, f, ensure_ascii=False, indent=2)
            print(f"Saved asin_to_text: {len(asin_to_text)} -> {asin_to_text_path}")

        if 'image' in need_to_load:
            asin_to_image = {asin: url for asin, url in loaded_image.items() if asin in id2item.values()}
            with open(asin_to_image_path, 'w', encoding='utf-8') as f:
                json.dump(asin_to_image, f, ensure_ascii=False, indent=2)
            print(f"Saved asin_to_image: {len(asin_to_image)} -> {asin_to_image_path}")

    print(f'Load {embedding_model} model')
    device = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'
    model = AutoModel.from_pretrained(
        embedding_model,
        torch_dtype='float16',
        trust_remote_code=True,
    )
    model = model.to(device)

    # 저장 디렉터리 생성
    embedding_dir = os.path.join(save_dir, f'{embedding_model_saved_name}')
    os.makedirs(embedding_dir, exist_ok=True)

    # text embeddings
    generate_and_save_embeddings(
        embedding_dir, model, asin_to_text, asin_to_image, batch_size,
        modality='text', desc='Text', id2item=id2item, filename=f'{embedding_model_saved_name}_text'
    )

    # image embeddings
    generate_and_save_embeddings(
        embedding_dir, model, asin_to_text, asin_to_image, batch_size,
        modality='image', desc='Image', id2item=id2item, filename=f'{embedding_model_saved_name}_image'
    )

    # fused embeddings
    generate_and_save_embeddings(
        embedding_dir, model, asin_to_text, asin_to_image, batch_size,
        modality='text_with_image', desc='Fused', id2item=id2item, filename=f'{embedding_model_saved_name}_fused'
    )


if __name__ == '__main__':
    main()
