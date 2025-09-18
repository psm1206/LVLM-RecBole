import re, html, warnings
import pandas as pd
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)


def clean_metadata(metadata_df: pd.DataFrame) -> pd.DataFrame:
    metadata_df = metadata_df.fillna('')
    str_cols = metadata_df.select_dtypes(include='object').columns

    _CSS_NEWLINE_RE = re.compile(r'\n')
    _SPACE_RE = re.compile(r'[ \t]{2,}')
    _QUOTES_RE = re.compile(r"[\"'‘’“”]")

    def full_clean(x):
        def _clean_str(s: str) -> str:
            s = html.unescape(s)
            s = BeautifulSoup(s, 'html.parser').get_text(separator=' ')
            s = _CSS_NEWLINE_RE.sub(' ', s)
            s = s.replace('\xa0', '')
            s = _SPACE_RE.sub(' ', s)
            s = _QUOTES_RE.sub('', s)

            return s.strip()

        return _clean_str(x) if isinstance(x, str) else x

    def filter_list(x):
        if isinstance(x, list):
            return [e for e in x if e]
        
        return x
    
    def filter_string(x):
        parts = [e.strip() for e in x.split(',')]
        filtered = [e for e in parts if e]
        
        return ', '.join(filtered)
        
    for col in str_cols:
        metadata_df[col] = metadata_df[col].apply(filter_list)
        metadata_df[col] = metadata_df[col].apply(lambda lst: '' if isinstance(lst, list) and not len(lst) else lst)
        metadata_df[col] = metadata_df[col].apply(lambda lst: ', '.join(lst[:-1]) + lst[-1] if isinstance(lst, list) and len(lst) else lst)
        metadata_df[col] = metadata_df[col].apply(full_clean)
        metadata_df[col] = metadata_df[col].apply(filter_string)

    for col in str_cols:
        metadata_df[col] = metadata_df[col].apply(html.unescape)

    return metadata_df


def generate_item_flatten_text_list(metadata_df, selected_features): # main function
    item_metadata_dict = metadata_df.set_index('asin').to_dict(orient='index')
    features = set(metadata_df.columns)
    
    item_metatext_dict = {}
    
    for item_id, meta in item_metadata_dict.items():
        text = ""
        for feat in selected_features[:-1]:
            if (feat not in features) or (not meta.get(feat)):
                text += f"{feat}: None ; "
            else:
                text += f"{feat}: {meta.get(feat)} ; "
        
        last_feat = selected_features[-1]
        if last_feat not in features:
            text += f"{last_feat}: None"
        else:
            text += f"{last_feat}: {meta.get(last_feat)}"
        
        item_metatext_dict[item_id] = text

    return item_metatext_dict


def get_unique_words(metadata_df, feature_list):
    unique_words = set()
    
    for _, item_meta in metadata_df.iterrows():
        for col in feature_list:
            text = str(item_meta.get(col, ""))

            for chunk in text.split(', '):
                for word in chunk.split():
                    word = word.lower().strip()
                    word = re.sub(r'[^a-z0-9]', '', word)
                    if not word:
                        continue

                    unique_words.add(word)

    return unique_words


def get_word2id_dict(unique_words):
    word2id_dict = {word: idx for idx, word in enumerate(unique_words)}

    return word2id_dict
