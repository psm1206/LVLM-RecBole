import os, random, argparse, html, re
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='beauty', help='beauty, sports, toys')
    parser.add_argument('--raw_path', type=str, default='raw/data')
    parser.add_argument('--save_path', type=str, default='raw/processed')
    parser.add_argument('--num_cores', type=int, default=10)
    parser.add_argument('--gpu_id', type=int, default=0, help='ID of running GPU')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')

    parser.add_argument('--embedding_model', type=str, default='Alibaba-NLP/gme-Qwen2-VL-2B-Instruct', help='Embedding encoder')
    parser.add_argument('--generative_model', type=str, default='Qwen/Qwen2-VL-2B-Instruct', help='Generative decoder')
    
    return parser.parse_args()


def seed_everything(seed=42):
    random.seed(seed)                        # Python random
    np.random.seed(seed)                     # NumPy
    os.environ["PYTHONHASHSEED"] = str(seed) # Python hash-based ops

    import torch
    torch.manual_seed(seed)              # PyTorch CPU
    torch.cuda.manual_seed(seed)         # PyTorch GPU
    torch.cuda.manual_seed_all(seed)     # Multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sample_users(rating_df, user_field, ratio=0.2):
    users = rating_df[user_field].unique()
    n_sample = max(1, int(len(users) * ratio))

    sampled_users = np.random.choice(users, size=n_sample, replace=False)
    
    return rating_df[rating_df[user_field].isin(sampled_users)].reset_index(drop=True)


def k_core_filtering(rating_df: pd.DataFrame, num_cores, user_field, item_field):
    while True:
        user_counts = rating_df[user_field].value_counts()
        valid_users = user_counts[user_counts >= num_cores].index
        rating_df = rating_df[rating_df[user_field].isin(valid_users)].reset_index(drop=True)

        item_counts = rating_df[item_field].value_counts()
        valid_items = item_counts[item_counts >= num_cores].index
        rating_df = rating_df[rating_df[item_field].isin(valid_items)].reset_index(drop=True)
        
        if len(user_counts) == len(valid_users) and len(item_counts) == len(valid_items):
            break

    return rating_df


def train_valid_test_split(rating_df, user_field, valid_ratio=0.1, test_ratio=0.2):
    train_list, valid_list, test_list = [], [], []

    for _, grp in rating_df.groupby(user_field):
        grp = grp.sample(frac=1, random_state=42)
        
        n = len(grp)
        n_valid, n_test = max(int(n * valid_ratio), 1), max(int(n * test_ratio), 1)
        n_train = n - n_valid - n_test
        
        train_list.append(grp.iloc[:n_train])
        valid_list.append(grp.iloc[n_train:n_train + n_valid])
        test_list.append(grp.iloc[n_train + n_valid:])

    train_rating_df, valid_rating_df, test_rating_df = pd.concat(train_list).reset_index(drop=True), pd.concat(valid_list).reset_index(drop=True), pd.concat(test_list) .reset_index(drop=True)

    return train_rating_df, valid_rating_df, test_rating_df


def get_amazon2018_rating_and_metadata_files(raw_path, dataset):
    root_path, dataset_full_name = os.getcwd(), amazon_dataset2fullname[dataset]
    file_prefix = os.path.join(root_path, raw_path, dataset)
    
    rating_file_path, metadata_file_path = os.path.join(file_prefix, f'{dataset_full_name}.json.gz'), os.path.join(file_prefix, f'meta_{dataset_full_name}.json.gz')

    return rating_file_path, metadata_file_path


def get_amazon2023_rating_and_metadata_files(raw_path, dataset):
    root_path, dataset_full_name = os.getcwd(), amazon_dataset2fullname[dataset]
    file_prefix = os.path.join(root_path, raw_path, dataset)
    
    rating_file_path, metadata_file_path = os.path.join(file_prefix, f'{dataset_full_name}.jsonl.gz'), os.path.join(file_prefix, f'meta_{dataset_full_name}.jsonl.gz')

    return rating_file_path, metadata_file_path


amazon_dataset2fullname = {
    'games': 'Video_Games',
    'toys': 'Toys_and_Games',
    'office': 'Office_Products',
    'beauty': 'Beauty',
    'clothing': 'Clothing_Shoes_and_Jewelry',
    'movies': 'Movies_and_TV',
    'books': 'Books',
    'sports': 'Sports_and_Outdoors',
    'scientifics': 'Industrial_and_Scientific',
    'home': 'Home_and_Kitchen',
}

amazon2023_dataset2main_category = {
    'games': 'Video Games',
    'toys': 'Toys & Games',
    'office': 'Office Products',

    'movies': 'Movies & TV',
    'books': 'Books',
    'sports': 'Sports & Outdoors',
    'scientifics': 'Industrial_and_Scientific',
}
