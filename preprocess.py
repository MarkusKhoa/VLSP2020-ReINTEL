import re
import torch
import json

stop_words_file = "./vietnamese-stopwords.txt"
with open(stop_words_file) as f:
    vnmese_stopwords = f.read().splitlines()

def normalizePost(post, tweet_tokenizer, vncorenlp, is_preprocessing = True,
                  use_segment=False, remove_punc_stopword=False, lowercase_opt=False,
                  truncation_method="head_only", length=256):
    if is_preprocessing:
      post = post.strip()
      URL_pattern = r"(?:http?s://|www.)[^\"]+"
      hashtag_pattern = r"#\w+"

      post = re.sub(URL_pattern, "", post)
      post = re.sub(hashtag_pattern, "hashtag", post)
      post = re.sub("\.+",".", post)
      # post = re.sub("#\s+", " ", post)
      post = re.sub("\*+", " ", post)
      post = re.sub("\$+", "đô ", post)
      post = re.sub("-{2,}", "", post)
      post = re.sub("\@+", "", post)
      post = re.sub("\[[0-9]?[0-9]]", " : dẫn_chứng ", post)

    post = post.strip()
    if lowercase_opt:
      post = post.lower()
    tokens = tweet_tokenizer.tokenize(post.replace("’", "'").replace("…", "..."))
    
    post = " ".join(tokens)
    if use_segment:
        tokens = vncorenlp.tokenize(post.replace("’", "'").replace("…", "..."))
        tokens = [t for ts in tokens for t in ts]
    normPost = " ".join(tokens)

    if remove_punc_stopword:
      tokens = [t for t in normPost if not t in vnmese_stopwords]
    normPost = " ".join(tokens)

    normPost = re.sub(r",([0-9]{2,4}) , ([0-9]{2,4})", r",\1,\2", normPost)
    normPost = re.sub(r"([0-9]{1,3}) / ([0-9]{2,4})", r"\1/\2", normPost)
    normPost = re.sub(r"([0-9]{1,3})- ([0-9]{2,4})", r"\1-\2", normPost)
    
    if use_segment:
        normPost = normPost.replace('< url >', '<url>')
        normPost = re.sub(r"# (\w+)", r'#\1', normPost)
    if truncation_method == "head_only":
      normPost = " ".join(normPost.split(" ")[:length])
    if truncation_method == "tail_only":
      normPost = " ".join(normPost.split(" ")[-length:])
    if truncation_method == "head_tail":
      normPost = " ".join(normPost.split(" ")[:int(length*0.25)]) + " " +  " ".join(normPost.split(" ")[-int(length*0.75):])

    replace_list = json.load(open("/content/drive/MyDrive/VLSP-Fake-News-Detection/replace_list.txt"))
    for k, v in replace_list.items():
        normPost = normPost.replace(k, v)
    return normPost

def convert_tokens_to_ids(texts, tokenizer, max_seq_length=256, labels=None):
    input_ids, attention_masks = [], []
    for text in texts:
        inputs = tokenizer.encode_plus(text, padding='max_length', max_length=max_seq_length, truncation=True)
        input_ids.append(inputs['input_ids'])
        attention_masks.append(inputs['attention_mask'])

    if labels is not None:
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(attention_masks, dtype=torch.long), torch.tensor(
            labels, dtype=torch.long)
    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(attention_masks, dtype=torch.long)