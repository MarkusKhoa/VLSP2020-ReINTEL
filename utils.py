from vncorenlp import VnCoreNLP
from nltk.tokenize import TweetTokenizer
from pandas import DataFrame

import re
import json

def pre_process(post, tweet_tokenizer, rdrsegmenter, use_segment,
                lowercase_opt, truncation_method = "head_tail", length = 256):
    
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
        tokens = rdrsegmenter.tokenize(post.replace("’", "'").replace("…", "..."))
        tokens = [t for ts in tokens for t in ts]
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