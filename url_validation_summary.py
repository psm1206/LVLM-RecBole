#!/usr/bin/env python3
"""
URL 검증 결과 요약 스크립트
여러 데이터셋의 URL 검증 결과를 종합하여 요약 보고서를 생성합니다.
"""

import os
import json
import pandas as pd
from typing import Dict, List
import argparse
from datetime import datetime


def load_validation_results(dataset_name: str) -> Dict:
    """검증 결과 파일을 로드합니다."""
    result_file = f"url_validation_results_{dataset_name}.json"
    if os.path.exists(result_file):
        with open(result_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def analyze_dataset(dataset_name: str) -> Dict:
    """데이터셋의 URL 검증 결과를 분석합니다."""
    results = load_validation_results(dataset_name)
    if not results:
        return None
    
    # 기본 통계
    stats = {
        'dataset': dataset_name,
        'total_asins': results['total_asins'],
        'asins_with_url': results['asins_with_url'],
        'asins_without_url': results['asins_without_url'],
        'url_coverage_rate': results['url_coverage_rate'],
        'validated_urls': results['validated_urls'],
        'valid_urls': results['valid_urls'],
        'invalid_urls': results['invalid_urls'],
        'validation_success_rate': results['validation_success_rate'],
        'validation_time': results['validation_time']
    }
    
    # 오류 유형별 분석
    error_types = {}
    if 'detailed_results' in results:
        for asin, data in results['detailed_results'].items():
            if not data['is_valid']:
                error_msg = data['message']
                if 'HTTP 404' in error_msg:
                    error_types['404 Not Found'] = error_types.get('404 Not Found', 0) + 1
                elif 'Timeout' in error_msg:
                    error_types['Timeout'] = error_types.get('Timeout', 0) + 1
                elif 'Connection Error' in error_msg:
                    error_types['Connection Error'] = error_types.get('Connection Error', 0) + 1
                elif 'Not image' in error_msg:
                    error_types['Not Image'] = error_types.get('Not Image', 0) + 1
                else:
                    error_types['Other'] = error_types.get('Other', 0) + 1
    
    stats['error_breakdown'] = error_types
    return stats


def generate_summary_report(datasets: List[str], output_file: str = None):
    """여러 데이터셋의 결과를 종합하여 요약 보고서를 생성합니다."""
    all_stats = []
    
    for dataset in datasets:
        stats = analyze_dataset(dataset)
        if stats:
            all_stats.append(stats)
    
    if not all_stats:
        print("분석할 수 있는 검증 결과가 없습니다.")
        return
    
    # DataFrame으로 변환
    df = pd.DataFrame(all_stats)
    
    # 보고서 내용을 문자열로 생성
    report_lines = []
    
    report_lines.append("=" * 80)
    report_lines.append("URL 검증 결과 종합 보고서")
    report_lines.append(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # 기본 통계 테이블
    report_lines.append("📊 기본 통계")
    report_lines.append("-" * 80)
    report_lines.append(f"{'데이터셋':<12} {'전체 ASIN':<10} {'URL 보유':<10} {'URL 없음':<10} {'보유율(%)':<10} {'검증 성공률(%)':<15}")
    report_lines.append("-" * 80)
    
    for _, row in df.iterrows():
        report_lines.append(f"{row['dataset']:<12} {row['total_asins']:<10} {row['asins_with_url']:<10} {row['asins_without_url']:<10} "
                           f"{row['url_coverage_rate']:<10.2f} {row['validation_success_rate']:<15.2f}")
    
    report_lines.append("")
    
    # 전체 요약
    total_asins = df['total_asins'].sum()
    total_with_url = df['asins_with_url'].sum()
    total_without_url = df['asins_without_url'].sum()
    total_validated = df['validated_urls'].sum()
    total_valid = df['valid_urls'].sum()
    total_invalid = df['invalid_urls'].sum()
    
    report_lines.append("📈 전체 요약")
    report_lines.append("-" * 80)
    report_lines.append(f"전체 ASIN 개수: {total_asins:,}")
    report_lines.append(f"이미지 URL이 있는 ASIN: {total_with_url:,} ({total_with_url/total_asins*100:.2f}%)")
    report_lines.append(f"이미지 URL이 없는 ASIN: {total_without_url:,} ({total_without_url/total_asins*100:.2f}%)")
    report_lines.append(f"검증된 URL 개수: {total_validated:,}")
    report_lines.append(f"접속 가능한 URL: {total_valid:,} ({total_valid/total_validated*100:.2f}%)")
    report_lines.append(f"접속 불가능한 URL: {total_invalid:,} ({total_invalid/total_validated*100:.2f}%)")
    report_lines.append("")
    
    # 오류 유형별 분석
    report_lines.append("🚨 오류 유형별 분석")
    report_lines.append("-" * 80)
    
    all_error_types = {}
    for _, row in df.iterrows():
        for error_type, count in row['error_breakdown'].items():
            all_error_types[error_type] = all_error_types.get(error_type, 0) + count
    
    if all_error_types:
        sorted_errors = sorted(all_error_types.items(), key=lambda x: x[1], reverse=True)
        report_lines.append(f"{'오류 유형':<20} {'개수':<10} {'비율(%)':<10}")
        report_lines.append("-" * 40)
        for error_type, count in sorted_errors:
            percentage = count / total_invalid * 100 if total_invalid > 0 else 0
            report_lines.append(f"{error_type:<20} {count:<10} {percentage:<10.2f}")
    else:
        report_lines.append("접속 불가능한 URL이 없습니다.")
    
    report_lines.append("")
    
    # 데이터셋별 상세 정보
    report_lines.append("📋 데이터셋별 상세 정보")
    report_lines.append("-" * 80)
    
    for _, row in df.iterrows():
        report_lines.append(f"\n🔍 {row['dataset'].upper()} 데이터셋:")
        report_lines.append(f"  • 전체 ASIN: {row['total_asins']:,}개")
        report_lines.append(f"  • URL 보유 ASIN: {row['asins_with_url']:,}개 ({row['url_coverage_rate']:.2f}%)")
        report_lines.append(f"  • URL 없음 ASIN: {row['asins_without_url']:,}개")
        report_lines.append(f"  • 검증된 URL: {row['validated_urls']:,}개")
        report_lines.append(f"  • 검증 통과한 URL: {row['valid_urls']:,}개 ({row['validation_success_rate']:.2f}%)")
        report_lines.append(f"  • 검증 실패한 URL: {row['invalid_urls']:,}개")
        report_lines.append(f"  • 검증 소요 시간: {row['validation_time']:.2f}초")
        
        if row['error_breakdown']:
            report_lines.append("  • 오류 유형:")
            for error_type, count in row['error_breakdown'].items():
                report_lines.append(f"    - {error_type}: {count}개")
        else:
            report_lines.append("  • 모든 URL이 정상적으로 접속 가능")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("보고서 생성 완료")
    report_lines.append("=" * 80)
    
    # 보고서 내용을 문자열로 결합
    report_content = "\n".join(report_lines)
    
    # 콘솔에 출력
    print(report_content)
    
    # 파일로 저장
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"url_validation_report_{timestamp}.txt"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"\n📄 보고서가 '{output_file}' 파일로 저장되었습니다.")


def main():
    parser = argparse.ArgumentParser(description='URL 검증 결과 요약 스크립트')
    parser.add_argument('--datasets', nargs='+', default=['beauty', 'clothing', 'home', 'sports', 'toys'], 
                       help='분석할 데이터셋 목록')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='출력 파일명 (지정하지 않으면 타임스탬프가 포함된 파일명 자동 생성)')
    
    args = parser.parse_args()
    
    # 실제로 검증 결과 파일이 있는 데이터셋만 필터링
    available_datasets = []
    for dataset in args.datasets:
        result_file = f"url_validation_results_{dataset}.json"
        if os.path.exists(result_file):
            available_datasets.append(dataset)
        else:
            print(f"⚠️  {dataset} 데이터셋의 검증 결과 파일을 찾을 수 없습니다.")
    
    if available_datasets:
        generate_summary_report(available_datasets, args.output)
    else:
        print("분석할 수 있는 검증 결과 파일이 없습니다.")


if __name__ == '__main__':
    main()
