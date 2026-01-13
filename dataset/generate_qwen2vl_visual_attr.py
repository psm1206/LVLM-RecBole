import os
import json
import ast
import pickle
import gc
import time
import pandas as pd
from typing import Dict, List, Tuple

import requests
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel
from transformers import Qwen2VLForConditionalGeneration, Qwen3VLForConditionalGeneration, AutoProcessor
from transformers.utils.versions import require_version
from qwen_vl_utils import process_vision_info
 

from utils.dataset import parse_args, seed_everything, amazon_dataset2fullname
from utils.text import clean_metadata


# require_version(
#     "transformers<4.52.0",
#     "The remote code has some issues with transformers>=4.52.0, please downgrade: pip install transformers==4.51.3"
# )
# transformers 4.51.0 -> Qwen2-VL
# transformers 4.57.0 -> Qwen3-VL


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

    return asin_to_text, asin_to_title, asin_to_image


def load_id2item(dataset_name: str) -> Dict[str, str]:
    id_map_path = os.path.join(dataset_name, 'id_map.json')
    with open(id_map_path, 'r') as f:
        obj = json.load(f)
    return obj['id2item']


def batched(iterable: List, batch_size: int) -> List[List]:
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]


def generate_text_description_from_image_with_vlm(asin_to_image, asin_to_title, vlm_model_name, device):
    """
    Generated text description is used for image2text embedding.
    Return format: {asin: {caption: text description, title: title, image_url: image url}}
    """
    asin_to_caption: Dict[str, str] = {}
    error_count = 0
    
    PROMPT_TEMPLATE = (
        "You are a helpful assistant.\n"
        "Given an Amazon product's title and its image, please provide a detailed, visually grounded description of the product "
        "that would help someone decide whether to purchase it. "
        "Focus on the product's appearance, features, and any other visually informative aspects. "
        "Do not mention the product's title in your answer. "
        "This product's title is: {title}\n"
        "Assistant:"
    )

    vlm_model = Qwen2VLForConditionalGeneration.from_pretrained(
        vlm_model_name, torch_dtype="auto", device_map=device
    )
    
    processor = AutoProcessor.from_pretrained(vlm_model_name)

    asins_with_image = [asin for asin, url in asin_to_image.items() if isinstance(url, str) and url.strip()]
    
    for asin in tqdm(asins_with_image, desc='Image2text'):
        url = asin_to_image[asin]
        title = asin_to_title[asin]
        
        try:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": url},
                    {"type": "text", "text": PROMPT_TEMPLATE.format(title=title)},
                ],
            }]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            img_inp, vid_inp = process_vision_info(messages)
            inputs = processor(text=text, images=img_inp, videos=vid_inp, return_tensors="pt")
            inputs = inputs.to(device)
            with torch.inference_mode():
                out_ids = vlm_model.generate(**inputs, max_new_tokens=128)
            out_ids_trimmed = out_ids[0][len(inputs.input_ids[0]):]
            caption = processor.decode(out_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            asin_to_caption[asin] = {
                'caption': caption,
                'title': title,
                'image_url': url,
            }
        except Exception as individual_error:
            error_count += 1
            print(f"Error generating caption for {asin}: {str(individual_error)}")
            continue

        # memory management (stability for large batches)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_items = max(1, len(asins_with_image))
    print(f"Image2text error count: {error_count}")
    print(f"Image2text error rate: {error_count / total_items * 100:.2f}%")

    return asin_to_caption

def generate_visual_attributes_with_vlm(asin_to_image, asin_to_title, asin_to_description, vlm_model_name, device):
    """
    Generated visual attributes are used for text embedding.
    Return format: {asin: {visual_style: visual style, material: material, purpose: purpose, usage_context: usage context}}
    """
    asin_to_visual_attr: Dict[str, str] = {}
    error_count = 0
    
    PROMPT_TEMPLATE = (
        "You are a professional product visual analyst. Your goal is to extract discriminative physical features from the image to distinguish this product from others. "
        "Reference the Title and Description for context, but prioritize what is actually visible in the image.\n\n"
        "### Guidelines:\n"
        "1. Visual Evidence First: For 'Visible On-Product Text', only transcribe words you can clearly read in the image. If the text is illegible or not present, state 'None'. Do not assume text exists just because it is in the Title.\n"
        "2. Fine-Grained Description: Avoid generic terms. Use specific descriptors for colors (e.g., champagne gold, matte teal), shapes (e.g., tapered cylinder, square-shouldered bottle), and textures.\n"
        "3. Conciseness: Use only 1-4 words per attribute value.\n"
        "4. Handling Missing Info: If a symbol or text is not clearly visible, you must state 'None'.\n\n"
        "### Input Information:\n"
        "- Title: {title}\n"
        "- Description: {description}\n\n"
        "### Attributes to Extract:\n"
        "1. Color & Aesthetic Tone: Specific colors, gradients, or overall tone.\n"
        "2. Structural Shape: Detailed silhouette including cap, lid, or applicator type.\n"
        "3. Surface Finish: Material texture and light interaction (e.g., frosted, reflective, matte).\n"
        "4. Container Mechanism: How the product is opened or dispensed.\n"
        "5. Visible On-Product Text: Brand or product names clearly legible in the image. If none, 'None'.\n"
        "6. Distinctive Icons: Visible logos, emblems, or unique patterns. If none, 'None'.\n\n"
        "### Output Format:\n"
        "Color & Aesthetic Tone: <value>\n"
        "Structural Shape: <value>\n"
        "Surface Finish: <value>\n"
        "Container Mechanism: <value>\n"
        "Visible On-Product Text: <value>\n"
        "Distinctive Icons: <value>\n\n"
        "Assistant:"
    )

    if 'Qwen3-VL' in vlm_model_name:
        vlm_model = Qwen3VLForConditionalGeneration.from_pretrained(
            vlm_model_name, torch_dtype="auto", device_map=device
        )
    elif 'Qwen2-VL' in vlm_model_name:
        vlm_model = Qwen2VLForConditionalGeneration.from_pretrained(
            vlm_model_name, torch_dtype="auto", device_map=device
        )
    else:
        raise ValueError(f"Invalid model name: {vlm_model_name}")
    print(f"Loaded {vlm_model_name} model")
    
    processor = AutoProcessor.from_pretrained(vlm_model_name)

    asins_with_image = [asin for asin, url in asin_to_image.items() if isinstance(url, str) and url.strip()]
    for asin in tqdm(asins_with_image, desc='Extracting Visual Attributes'):
        url = asin_to_image[asin]
        title = asin_to_title[asin]
        description = asin_to_description[asin]
        
        try:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": url},
                    {"type": "text", "text": PROMPT_TEMPLATE.format(title=title, description=description)},
                ],
            }]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            img_inp, vid_inp = process_vision_info(messages)
            inputs = processor(text=text, images=img_inp, videos=vid_inp, return_tensors="pt").to(device)
            
            with torch.inference_mode():
                out_ids = vlm_model.generate(**inputs, max_new_tokens=80)
            
            out_ids_trimmed = out_ids[0][len(inputs.input_ids[0]):]
            output = processor.decode(out_ids_trimmed, skip_special_tokens=True).strip()
            
            attr = {'color_aesthetic': 'N/A', 'physical_shape': 'N/A', 'material_finish': 'N/A', 
                    'form_container_type': 'N/A', 'visible_on_product_text': 'N/A', 'distinctive_marks_symbols': 'N/A'}
            for line in output.split('\n'):
                line = line.strip()
                if line.lower().startswith('color & aesthetic tone:'):
                    attr['color_aesthetic'] = line.split(':', 1)[1].strip()
                elif line.lower().startswith('structural shape:'):
                    attr['physical_shape'] = line.split(':', 1)[1].strip()
                elif line.lower().startswith('surface finish:'):
                    attr['material_finish'] = line.split(':', 1)[1].strip()
                elif line.lower().startswith('container mechanism:'):
                    attr['form_container_type'] = line.split(':', 1)[1].strip()
                elif line.lower().startswith('visible on-product text:'):
                    attr['visible_on_product_text'] = line.split(':', 1)[1].strip()
                elif line.lower().startswith('distinctive icons:'):
                    attr['distinctive_marks_symbols'] = line.split(':', 1)[1].strip()

            formatted_text = (
                f"Color & Aesthetic Tone is {attr['color_aesthetic']}; "
                f"Structural Shape is {attr['physical_shape']}; "
                f"Surface Finish is {attr['material_finish']}; "
                f"Container Mechanism is {attr['form_container_type']}; "
                f"Visible On-Product Text is {attr['visible_on_product_text']}; "
                f"Distinctive Icons is {attr['distinctive_marks_symbols']}."
            )
            
            asin_to_visual_attr[asin] = formatted_text

        except Exception as e:
            error_count += 1
            print(f"Error for {asin}: {str(e)}")
            continue

        # print first 10 items of asin_to_visual_attr
        if len(asin_to_visual_attr) <= 10 and asin in asin_to_visual_attr:
            print(f"Visual attr for {asin}: {asin_to_visual_attr[asin]}")

        gc.collect()
        torch.cuda.empty_cache()

    del vlm_model
    return asin_to_visual_attr    


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


# Determine dims using any available vector per modality
def pick_dim(d) -> int:
    for v in d.values():
        return int(v.shape[-1])
    return 0


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
        embeds = generate_embeddings_with_gme(
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
    generative_model = args.generative_model
    batch_size = args.batch_size
    gpu_id = args.gpu_id
    device = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'

    short_name_dict = {'Qwen/Qwen2-VL-2B-Instruct': 'qwen2vl2b',
                        'Qwen/Qwen2-VL-7B-Instruct': 'qwen2vl7b',
                        'Qwen/Qwen3-VL-2B-Instruct': 'qwen3vl2b',
                        'Qwen/Qwen3-VL-4B-Instruct': 'qwen3vl4b',
                        'Qwen/Qwen3-VL-8B-Instruct': 'qwen3vl8b',
                        'Alibaba-NLP/gme-Qwen2-VL-2B-Instruct': 'gme_qwen2vl2b',
                        'Alibaba-NLP/gme-Qwen2-VL-7B-Instruct': 'gme_qwen2vl7b'}

    embedding_model_saved_name = short_name_dict[embedding_model]
    generative_model_saved_name = short_name_dict[generative_model]

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

    # image attributes from image
    image_attr_path = os.path.join(save_dir, f'{generative_model_saved_name}_image_attr_v3.json')
    if os.path.exists(image_attr_path):
        print(f"Image attr already exist: {image_attr_path}")
        asin_to_visual_attr = json.load(open(image_attr_path, 'r', encoding='utf-8'))
    else:
        asin_to_visual_attr = generate_visual_attributes_with_vlm(
            asin_to_image, asin_to_title, asin_to_description, generative_model, device
        )
        with open(image_attr_path, 'w', encoding='utf-8') as f:
            json.dump(asin_to_visual_attr, f, ensure_ascii=False, indent=2)
        print(f"Saved Image attr captions: {len(asin_to_visual_attr)} -> {image_attr_path}")

    # Load embedding model
    print(f'Load {embedding_model} model')
    model = AutoModel.from_pretrained(
        embedding_model,
        torch_dtype='float16',
        trust_remote_code=True,
    )
    model = model.to(device)

    embedding_dir = os.path.join(save_dir, f'{embedding_model_saved_name}')
    os.makedirs(embedding_dir, exist_ok=True)

    # Image attribute embeddings
    generate_and_save_embeddings(
        embedding_dir, model, asin_to_visual_attr, {}, {}, batch_size,
        modality='text', desc='Image attr', id2item=id2item, filename='image_attr_v3'
    )

    # Fused text + image attribute embeddings
    generate_and_save_embeddings(
        embedding_dir, model, asin_to_text, asin_to_visual_attr, {}, batch_size,
        modality='text_with_text', desc='Text+Image attr', id2item=id2item, filename='text_with_image_attr_v3'
    )


if __name__ == '__main__':
    main()
