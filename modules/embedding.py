
import pickle
import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BertModel, AutoModel
from functools import partial
import multiprocessing as mp
import numpy as np

from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import gc

def cleanse_word(word):
  clean_word = ""
  for char in word:
    if char in 'qwertyuiopasdfghjklzxcvbnm ':
      clean_word += char
  return clean_word

def merge_tokens(tokens, features): # unbatched tokens, features
  merged_tokens = []
  merged_features = []
  assert len(tokens) == len(features)
  for token, feature in zip(tokens, features):
    word = token.replace("##", "")
    
    if token.startswith("##"):
      merged_tokens[-1] += word
      merged_features[-1] += feature
    else:
      merged_tokens.append(word)
      merged_features.append(feature)
  assert len(merged_tokens) == len(merged_features)
  return merged_tokens, merged_features

def _create_word_dict(batch, keyword_set):
  word_vectors = defaultdict(list)
  merged_tokens, merged_features = merge_tokens(*batch)
  for feature, keyword in zip(merged_features, merged_tokens):
    keyword = cleanse_word(keyword)
    if keyword in keyword_set:
      word_vectors[keyword].append(feature.numpy(force=True)) # 문서 안에 한 단어 여러번 등장시 평균 임베딩으로 선택
  word_vectors = dict(word_vectors)
  for word, vecs in word_vectors.items():
    word_vectors[word] = np.average(vecs, axis=0)
  return word_vectors

def create_word_dicts(tokens, features, keyword_set, pool):
    features = features.cpu()
    fn = partial(_create_word_dict, keyword_set = keyword_set)
    word_vectors_list = pool.map(fn, zip(tokens, features))
    return word_vectors_list

def create_embedding(dataset, essential, tokenizer, bert):
  loader = DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=0)
  embedding = dict()
  doc_label_map = dict()
  keyword_set = set(essential['keywords'])
  with mp.Pool(5) as pool:
    with torch.no_grad():
      for docs in tqdm(loader):
        labels, indexed_tokens, segments_ids, num_docs, is_augmented = docs
        tokens = [tokenizer.convert_ids_to_tokens(ids) for ids in indexed_tokens]
        hidden_states = bert(indexed_tokens.cuda(), segments_ids.cuda())[2]
        token_features = torch.stack(hidden_states, dim=0).permute(1, 2, 0, 3) # dim -> batch, token, layer, features
        sum_vec = torch.sum(token_features[:, :, -4:, :], dim=2) # merged last 4 layers output -> batch, token, features
        word_dicts_list = create_word_dicts(tokens, sum_vec, keyword_set, pool)
        for label, num_doc, word_dict in zip(labels, num_docs, word_dicts_list):
          num_doc = int(num_doc)
          label = int(label)
          embedding[num_doc] = word_dict
          doc_label_map[num_doc] = label
  return embedding, doc_label_map


def generate_embeddings(args, essential, large_data, force=True):
  print("Generating Embeddings")
  dataset_name = args['dataset_name']
  train_set = essential['train_set']
  val_set = essential['val_set']
  test_set = essential['test_set']
  tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-tiny', model_max_length = 512) 

  # bert = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True).cuda()
  bert = AutoModel.from_pretrained("prajjwal1/bert-tiny", output_hidden_states = True).cuda()
  bert.eval()

  train_embedding, train_doc_label_map = create_embedding(train_set, essential, tokenizer, bert)
  
  os.makedirs(f"./data/{dataset_name}", exist_ok=True)
  embedding_path = f"data/{dataset_name}/embedding.pkl"
  if os.path.exists(embedding_path) and not force:
    with open(embedding_path, "rb") as f:
      val_embedding, test_embedding = pickle.load(f)
  else:
    val_embedding, _ = create_embedding(val_set, essential, tokenizer, bert)
    test_embedding, _ = create_embedding(test_set, essential, tokenizer, bert)
    pass
    
  gc.collect()
  torch.cuda.empty_cache()
  
  large_data['train_embedding'] = train_embedding
  large_data['val_embedding'] = val_embedding
  large_data['test_embedding'] = test_embedding
  large_data['train_doc_label_map'] = train_doc_label_map

  print("Words per document statistics")
  for emb in ["train_embedding"]:
    lengths = [len(w) for w in large_data[emb].values()]
    print(emb, np.mean(lengths), np.std(lengths))
    # plt.plot(lengths)
    # plt.savefig("./embedding.png")
    # plt.close()

def plot_embeddings(train_embedding, val_embedding, test_embedding):
  plt.clf()
  keyword_lengths = []
  for doc, keyword_features in train_embedding.items():
      num_keywords = len(keyword_features.keys())
      keyword_lengths.append((num_keywords, doc))
  keyword_lengths.sort(key=lambda x:x[0])
  plt.plot(range(len(keyword_lengths)), [k[0] for k in keyword_lengths], label='train')

  keyword_lengths = []
  for doc, keyword_features in val_embedding.items():
      num_keywords = len(keyword_features.keys())
      keyword_lengths.append((num_keywords, doc))
  keyword_lengths.sort(key=lambda x:x[0])
  plt.plot(range(len(keyword_lengths)), [k[0] for k in keyword_lengths], label='val')

  keyword_lengths = []
  for doc, keyword_features in test_embedding.items():
      num_keywords = len(keyword_features.keys())
      keyword_lengths.append((num_keywords, doc))
  keyword_lengths.sort(key=lambda x:x[0])
  plt.plot(range(len(keyword_lengths)), [k[0] for k in keyword_lengths], label='test')
  plt.legend()
  # plt.show()