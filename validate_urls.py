#!/usr/bin/env python3
"""
asin_to_image dictionary의 URL 검증 스크립트
- URL이 없는 ASIN 개수 확인
- URL이 존재하는 ASIN들의 이미지 접속 가능성 검증
"""

import os
import json
import ast
import pandas as pd
import requests
from typing import Dict, List, Tuple
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from dataset.utils.dataset import amazon_dataset2fullname
from dataset.utils.text import clean_metadata


def load_metadata_and_id2item(dataset_name: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """메타데이터와 id2item 매핑을 로드합니다."""
    dataset_full_name = amazon_dataset2fullname[dataset_name]
    meta_path = os.path.join('dataset', dataset_name, f'meta_{dataset_full_name}.json')
    asin_to_image = {}

    # 메타데이터 로드
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
                'imUrl': obj.get('imUrl', ''),
            })

    if not rows:
        return asin_to_image, {}

    df = pd.DataFrame(rows)
    df['imUrl'] = df['imUrl'].astype(str).apply(lambda x: x.strip())

    # 텍스트 전처리
    df = clean_metadata(df)

    for _, row in df.iterrows():
        asin = row['asin']
        image_url = row.get('imUrl', '')
        if isinstance(image_url, str) and image_url:
            asin_to_image[asin] = image_url

    # id2item 매핑 로드
    id_map_path = os.path.join('dataset', dataset_name, 'id_map.json')
    with open(id_map_path, 'r') as f:
        obj = json.load(f)
    id2item = obj['id2item']

    return asin_to_image, id2item


def check_url_validity(url: str, timeout: int = 10) -> Tuple[bool, str]:
    """URL의 유효성을 검사합니다."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36'
        }
        response = requests.head(url, headers=headers, timeout=timeout, allow_redirects=True)
        
        # 200-299 상태 코드이면서 Content-Type이 이미지인 경우
        if 200 <= response.status_code < 300:
            content_type = response.headers.get('content-type', '').lower()
            if 'image' in content_type:
                return True, f"OK (Status: {response.status_code}, Content-Type: {content_type})"
            else:
                return False, f"Not image (Status: {response.status_code}, Content-Type: {content_type})"
        else:
            return False, f"HTTP {response.status_code}"
    except requests.exceptions.Timeout:
        return False, "Timeout"
    except requests.exceptions.ConnectionError:
        return False, "Connection Error"
    except requests.exceptions.RequestException as e:
        return False, f"Request Error: {str(e)}"
    except Exception as e:
        return False, f"Unexpected Error: {str(e)}"


def validate_urls_parallel(urls: List[Tuple[str, str]], max_workers: int = 10) -> Dict[str, Tuple[bool, str]]:
    """병렬로 URL들을 검증합니다."""
    results = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Future 객체와 ASIN 매핑
        future_to_asin = {
            executor.submit(check_url_validity, url): asin 
            for asin, url in urls
        }
        
        for future in tqdm(as_completed(future_to_asin), total=len(urls), desc="URL 검증 중"):
            asin = future_to_asin[future]
            try:
                is_valid, message = future.result()
                results[asin] = (is_valid, message)
            except Exception as e:
                results[asin] = (False, f"Exception: {str(e)}")
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='URL 검증 스크립트')
    parser.add_argument('--dataset', type=str, required=True, help='데이터셋 이름 (예: beauty, clothing)')
    parser.add_argument('--max_workers', type=int, default=10, help='병렬 처리 스레드 수')
    parser.add_argument('--timeout', type=int, default=10, help='URL 요청 타임아웃 (초)')
    parser.add_argument('--sample_size', type=int, default=None, help='검증할 샘플 수 (None이면 전체)')
    
    args = parser.parse_args()
    
    print(f"데이터셋: {args.dataset}")
    print("=" * 50)
    
    # 메타데이터와 id2item 로드
    asin_to_image, id2item = load_metadata_and_id2item(args.dataset)
    
    # id2item에 있는 ASIN들만 필터링
    asin_to_image = {asin: url for asin, url in asin_to_image.items() if asin in id2item.values()}
    
    print(f"전체 ASIN 개수 (id2item 기준): {len(id2item)}")
    print(f"이미지 URL이 있는 ASIN 개수: {len(asin_to_image)}")
    print(f"이미지 URL이 없는 ASIN 개수: {len(id2item) - len(asin_to_image)}")
    print(f"이미지 URL 보유율: {len(asin_to_image) / len(id2item) * 100:.2f}%")
    print()
    
    if len(asin_to_image) == 0:
        print("검증할 URL이 없습니다.")
        return
    
    # 샘플링 (필요한 경우)
    urls_to_validate = list(asin_to_image.items())
    if args.sample_size and args.sample_size < len(urls_to_validate):
        import random
        random.seed(42)
        urls_to_validate = random.sample(urls_to_validate, args.sample_size)
        print(f"샘플링된 URL 개수: {len(urls_to_validate)}")
        print()
    
    # URL 검증
    print("URL 접속 가능성 검증 중...")
    start_time = time.time()
    
    results = validate_urls_parallel(urls_to_validate, max_workers=args.max_workers)
    
    end_time = time.time()
    print(f"검증 완료 (소요 시간: {end_time - start_time:.2f}초)")
    print()
    
    # 결과 분석
    valid_count = sum(1 for is_valid, _ in results.values() if is_valid)
    invalid_count = len(results) - valid_count
    
    print("=" * 50)
    print("검증 결과:")
    print(f"접속 가능한 URL: {valid_count}개 ({valid_count / len(results) * 100:.2f}%)")
    print(f"접속 불가능한 URL: {invalid_count}개 ({invalid_count / len(results) * 100:.2f}%)")
    print()
    
    # 접속 불가능한 URL들의 상세 정보
    if invalid_count > 0:
        print("접속 불가능한 URL들 (처음 20개):")
        invalid_urls = [(asin, message) for asin, (is_valid, message) in results.items() if not is_valid]
        for i, (asin, message) in enumerate(invalid_urls[:20]):
            url = asin_to_image[asin]
            print(f"{i+1:2d}. ASIN: {asin}")
            print(f"    URL: {url}")
            print(f"    오류: {message}")
            print()
        
        if len(invalid_urls) > 20:
            print(f"... 그리고 {len(invalid_urls) - 20}개 더")
    
    # 결과를 파일로 저장
    output_file = f"url_validation_results_{args.dataset}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'dataset': args.dataset,
            'total_asins': len(id2item),
            'asins_with_url': len(asin_to_image),
            'asins_without_url': len(id2item) - len(asin_to_image),
            'url_coverage_rate': len(asin_to_image) / len(id2item) * 100,
            'validated_urls': len(results),
            'valid_urls': valid_count,
            'invalid_urls': invalid_count,
            'validation_success_rate': valid_count / len(results) * 100 if len(results) > 0 else 0,
            'validation_time': end_time - start_time,
            'detailed_results': {
                asin: {
                    'url': asin_to_image[asin],
                    'is_valid': is_valid,
                    'message': message
                } for asin, (is_valid, message) in results.items()
            }
        }, f, ensure_ascii=False, indent=2)
    
    print(f"상세 결과가 {output_file}에 저장되었습니다.")


if __name__ == '__main__':
    main()
