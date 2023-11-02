import os
import random
import subprocess
import math
import torch
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import normalize
from functools import partial
import parmap
import pickle
import pandas as pd

from transformers import MarianMTModel, MarianTokenizer

def download(model_name):
  tokenizer = MarianTokenizer.from_pretrained(model_name)
  model = MarianMTModel.from_pretrained(model_name)
  return tokenizer, model




def load_pickle(path):
  with open(path, 'rb') as f:
    obj = pickle.load(f)
  return obj

def save_pickle(path, obj):
  with open(path, 'wb') as f:
    pickle.dump(obj, f, -1)

def pre_augmentation(args, essential):
  all_mode = 'all' in args['augment_method']
  augment_ratio = args['augment_ratio']
  print("Pre Augmentation")
  if ("EDA" in args['augment_method'] or all_mode) and args['augment_ratio'] > 0:
    print("Applying EDA")
    train_dataset = essential['train_dataset']
    eda_input = []
    for data in train_dataset:
        label = data['label']
        text = data['text']
        assert len(text) > 0
        eda_input.append((label, text))
    augment_amount = max(2, int(1.2*augment_ratio))
    save_pickle("cache/eda_input", eda_input)
    subprocess.run(f'python3 eda_nlp/code/augment.py --input="cache/eda_input" --output="cache/eda_output" --num_aug={augment_amount} --alpha_sr=0.1 --alpha_rd=0.1 --alpha_ri=0.1 --alpha_rs=0.1', check=True, shell=True)
    results = load_pickle("cache/eda_output")
    selection_amount = math.ceil(len(train_dataset)*augment_ratio)
    sampled = random.sample(results, selection_amount)
    for label, text in sampled:
      train_dataset.append({
        "label": int(label),
        "text": text,
        "augmented": True
      })
      
  if ('back' in args['augment_method'] or all_mode) and args['augment_ratio'] > 0:
    print("Applying Back Translation")
    train_dataset = essential['train_dataset']
    csv_path = f"cache/back_translated/{args['dataset_name']}_{args['data_ratio']}_{args['augment_ratio']}.csv"
    if not os.path.exists(csv_path):
      aug_input = []
      for _ in range(math.ceil(augment_ratio)):
        for data in train_dataset:
          label = data['label']
          text = data['text']
          if data['augmented']: continue
          assert len(text) > 0
          aug_input.append((label, text))
      sampled = aug_input
      augmented_data = []
      for label, text in tqdm(sampled):
        paraphrase = back_translate([text], 'en', 'fr')[0]
        augmented_data.append((label, paraphrase))
      df = pd.DataFrame(augmented_data, columns=["label", "text"])
      df.to_csv(csv_path, index=False)

    df = pd.read_csv(csv_path)
    for idx, row in df.iterrows():
      train_dataset.append({
        "label": row['label'],
        "text": row['text'],
        "augmented": True
      })
    

def translate(texts, model, tokenizer, language):
  """Translate texts into a target language"""
  # Format the text as expected by the model
  original_texts = texts

  # Tokenize (text to tokens)
  tokens = tokenizer.prepare_seq2seq_batch(original_texts, return_tensors='pt', padding=True)
  for k in tokens.keys():
    tokens[k] = tokens[k].cuda()

  # Translate
  translated = model.generate(**tokens)

  # Decode (tokens to text)
  translated_texts = tokenizer.batch_decode(translated, skip_special_tokens=True)

  return translated_texts

# download model for English -> French
tmp_lang_tokenizer, tmp_lang_model = download('Helsinki-NLP/opus-mt-en-fr')
# download model for French -> English
src_lang_tokenizer, src_lang_model = download('Helsinki-NLP/opus-mt-fr-en')
tmp_lang_model.cuda()
src_lang_model.cuda()

def back_translate(texts, language_src, language_dst):
  """Implements back translation"""
  # Translate from source to target language
  translated = translate(texts, tmp_lang_model, tmp_lang_tokenizer, language_dst)

  # Translate from target language back to source language
  back_translated = translate(translated, src_lang_model, src_lang_tokenizer, language_src)

  return back_translated
      

def _distance_sorting(target_vec, k, doc_words, vectors):
  distance = np.sum(np.square(vectors-target_vec), axis=1)
  merged = sorted(zip(doc_words, distance.tolist()), key=lambda x:x[1])
  return [m[0] for m in merged[:k]]# only (doc, word) list

def top_k_augmentation(args, essential, large_data):
    k = args['k']
    topk = defaultdict(dict)
    embedding = large_data['train_embedding']
    doc_words, vectors = zip(*[((doc, word), vec) for doc in embedding.keys() for word, vec in embedding[doc].items()])
    vectors = normalize(np.array(vectors))
    fn = partial(_distance_sorting, k=k, doc_words=doc_words, vectors=vectors)
    tasks = []
    for target_doc, word_vector in embedding.items():
        for target_word, target_vec in word_vector.items():
            tasks.append(target_vec)
    result = parmap.map(fn, tasks, pm_pbar=True, pm_processes=5)
    for (doc, word), vec in zip(doc_words, result):
      topk[doc][word] = vec
    topk = dict(topk)
    large_data['topk'] = topk
                    
def get_new_feature(args, doc_num, keyword, feature, large_data, train_type):
  if args['augment_method'] == 'kmeans':
    group_hashmap = large_data["kmeans_result"][train_type]["group_hashmap"]
    list_groups = large_data["kmeans_result"][train_type]["list_groups"]
    valid_centroid_map = large_data["kmeans_result"]["valid_centroid_map"]

    key = f"{doc_num}/{keyword}"
    cluster = group_hashmap[key]
    if valid_centroid_map[cluster] and len(list_groups[cluster]) > 1:
      return random.choice(list_groups[cluster])
    else:
      return (keyword, feature)
  elif args['augment_method'] == 'topk':
    assert train_type == 'train'
    embedding = large_data['train_embedding']
    topk = large_data['topk']
    (new_doc, new_keyword) = random.choice(topk[doc_num][keyword])
    feature = embedding[new_doc][new_keyword]
    return (keyword, feature)



def augment_keywords(args, doc_num, word_vector, large_data, change_ratio, train_type, augment_type):
  new_keyword_features = dict()
  assert augment_type in ["exchange", "insert", "delete"]
  word_candidates = list(word_vector.keys())
  augment_amount = max(math.ceil(len(word_candidates)*change_ratio), 1)
  augment_words = set(random.sample(word_candidates, augment_amount)) if len(word_candidates)>0 else set()
  for doc_word, feature in word_vector.items():
    if augment_type == 'exchange': # exchange word features
      if doc_word in augment_words: 
        new_word, new_feature = get_new_feature(args, doc_num, doc_word, feature, large_data, train_type)
        new_keyword_features[new_word] = new_feature
      else:
        new_keyword_features[doc_word] = word_vector[doc_word]
    elif augment_type == "insert": # insert new word features
      new_keyword_features[doc_word] = word_vector[doc_word]
      if doc_word in augment_words:
        new_word, new_feature = get_new_feature(args, doc_num, doc_word, feature, large_data, train_type)
        new_keyword_features[new_word] = new_feature
    elif augment_type == 'delete': # delete word features
      if doc_word in augment_words:
        pass
      else:
        new_keyword_features[doc_word] = word_vector[doc_word]
    else:
      raise Exception(f"Wrong augment type: {augment_type}")
  return new_keyword_features