from torch.utils.data import Dataset
from collections import defaultdict
from transformers import AutoTokenizer
import torch
import pickle
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import shutil
import math

from cuboid import augment_keywords, build_plain_cuboid, build_plain_interpolated, build_plain_cuboid_directly
from models import WordDecoder, Classifier

tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-tiny', model_max_length = 512) 
tokenizer_options = {
    "padding" : 'max_length', 
    "return_token_type_ids" : True, 
    "return_attention_mask" : False,
    "truncation" : True
}


def load_pickle(path):
  with open(path, 'rb') as f:
    obj = pickle.load(f)
  return obj

def save_pickle(path, obj):
  with open(path, 'wb') as f:
    pickle.dump(obj, f, -1)

class PreDataset(Dataset):
    def __init__(self, dataset, all_categories, keywords, features=False, data_count=0):
        self.labels = []
        self.all_categories = all_categories
        if features:
          self.features = []
        else:
          self.features = None
        self.document_numbers = []
        self.is_augmented = []
        cat_counts = defaultdict(int)
        for row in dataset:
          label = row['label']
          text = row['text']
          is_augmented = row['augmented']
          if len(text) ==0: 
            print("Empty text detected")
            continue
          
          is_empty = True
          for keyword in keywords:
            if keyword in text:
              is_empty = False
              break 
          if is_empty: continue
          self.is_augmented.append(is_augmented)
          if cat_counts[label]<data_count or data_count == 0:
            cat_counts[label]+=1
            self.labels.append(label)
            if features:
              indexed_tokens = tokenizer(text, **tokenizer_options)['input_ids']
              segments_ids = [1] * len(indexed_tokens)
              self.features.append((indexed_tokens, segments_ids))
        
        cat_counts = dict(cat_counts)
        print(cat_counts)
        self.document_numbers = list(range(len(self.labels)))

        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
      if self.features is not None:
        indexed_token, segments_id = self.features[idx]
        indexed_token = torch.tensor(indexed_token)
        segments_id = torch.tensor(segments_id)
      else:
        indexed_token = -999
        segments_id = -999
      label = torch.tensor(self.labels[idx])
      document_num = self.document_numbers[idx]
      is_augmented = self.is_augmented[idx]
      return label, indexed_token, segments_id, document_num, is_augmented


class GanDataset(Dataset):
  def __init__(self, args, essential, large_data, pre_dataset, dataset_type='train'):
    self.labels = []
    self.doc_nums = []
    self.is_augmented_list = []
    self.dataset_type = dataset_type

    self.essential = essential
    self.large_data = large_data
    print(f"Generating Gan Dataset")

    for label, _, _, doc_num, is_augmented in tqdm(pre_dataset):
      self.labels.append(label)
      self.doc_nums.append(doc_num)

  def __len__(self):
    return len(self.labels)
  
  def __getitem__(self, idx):
    label = self.labels[idx]
    cuboid = build_plain_cuboid(self.doc_nums[idx], self.essential, self.large_data, self.dataset_type)
    
    return label, cuboid

def build_compact_cuboid(doc_num, essential, large_data, train_type):
  embedding = large_data[train_type+'_embedding']
  keywords = essential['keywords']
  keywords_index_map = essential['keywords_index_map']
  feature_size = 128
  cuboid = torch.zeros(len(word_vector.keys()), feature_size, dtype=torch.float32, device='cuda')
  word_vector = embedding[doc_num]
  for keyword, feature in word_vector.items():
    row = keywords_index_map[keyword]
    cuboid[row] = torch.from_numpy(feature)
  return cuboid

class CuboidDataset(Dataset):
  def __init__(self, args, essential, large_data, train_type, pre_dataset):
    self.labels = []
    self.doc_nums = []
    self.is_augmented_list = []
    self.gan_augmented_list = []
    self.use_cuboid = args['use_cuboid']

    self.essential = essential
    self.large_data = large_data
    self.train_type = train_type
    self.augment_method = args['augment_method']
    print(f"Generating {train_type} CuboidDataset")

    for label, _, _, doc_num, is_augmented in tqdm(pre_dataset):
      if is_augmented and ("gan" in self.augment_method): continue
      self.labels.append(label)
      self.is_augmented_list.append(is_augmented)
      self.gan_augmented_list.append(False)
      self.doc_nums.append(doc_num)
    dataset_length = len(self.labels)

    if train_type == 'train' and ('gan' in self.augment_method) and args['augment_ratio'] > 0:
      self.augment_folder = "cache/augmented/"
      if os.path.exists(self.augment_folder): shutil.rmtree(self.augment_folder)
      os.makedirs(self.augment_folder)

      num_keywords = len(essential['keywords'])
      num_features = 128 # feature size
      num_category = len(essential['all_categories'])
      word_latent = args['word_latent']
      base_cuboid = large_data['base_cuboid']
      order_encoding = large_data['order_encoding']
      word_decoder = WordDecoder(num_keywords, num_category, num_features, order_encoding, base_cuboid, word_latent).cuda()
      word_decoder.load_state_dict(torch.load("./word_decoder.pt"))
      embedding = large_data['train_embedding']
      data_count = len(self.doc_nums)
      augment_ratio = args['augment_ratio']
      

      augment_amount = math.ceil(dataset_length*augment_ratio)
      augment_indices = random.sample(list(range(dataset_length))*math.ceil(augment_ratio), augment_amount)
      augmented_data_count = 0
      num_category = len(essential['all_categories'])
      keywords = essential['keywords']
      if args['line_augment_ratio'] > 0:
        word_groups = large_data['word_groups']

      with torch.no_grad():
        for i, idx in enumerate(tqdm(augment_indices)):
          new_doc_num = data_count+i
          label, _, _, doc_num, _ = pre_dataset[idx]
          ## Line Augmentation
          line = torch.zeros((args['num_keywords']), device='cuda')
          
          if random.random() < args['line_augment_ratio']:
            keys = list(embedding[doc_num].keys())
            select_counts = math.ceil(len(keys)*0.1) # 10% line augmentation
            line_method = random.choice(["insert", "delete", "exchange"])
            if line_method == 'insert':
              base_keys = random.sample(keys, select_counts)
              new_keys = [random.choice(word_groups[k]) for k in base_keys]
              remove_keys = set()
            elif line_method == 'exchange':
              base_keys = random.sample(keys, select_counts)
              new_keys = [random.choice(word_groups[k]) for k in base_keys]
              remove_keys = random.sample(keys, select_counts)
            else:
              remove_keys = random.sample(keys, select_counts)
              new_keys = set()

            for i, k in enumerate(keywords):
              if (k in keys and k not in remove_keys) or k in new_keys:
                line[i]=1
          else:
            keys = set(embedding[doc_num].keys())
            for i, k in enumerate(keywords):
              if k in keys: line[i]=1
        
          word_latent = torch.normal(0, 1, (1, args['num_keywords'], word_decoder.latent_size), device='cuda')
          new_cuboid = word_decoder(word_latent, label.cuda())[0]

          new_cuboid = new_cuboid * line.unsqueeze(1)
          new_word_vector = extract_word_vector(new_cuboid, line, keywords)
          save_pickle(self.augment_folder+str(new_doc_num), new_word_vector)
          
          self.labels.append(label)
          self.is_augmented_list.append(True)
          self.gan_augmented_list.append(True)
          self.doc_nums.append(new_doc_num)
          augmented_data_count+=1
      # large_data['train_embedding'] = embedding | new_embedding
      print(f"Total {augmented_data_count}/{len(augment_indices)} augmented")

  def __len__(self):
    return len(self.labels)
  
  def __getitem__(self, idx):
    label = self.labels[idx]
    is_augmented = self.is_augmented_list[idx]
    gan_augmented = self.gan_augmented_list[idx]
    doc_num = self.doc_nums[idx]
    if self.use_cuboid:
      if gan_augmented:
        word_vector = load_pickle(self.augment_folder+str(doc_num))
        cuboid = build_plain_cuboid_directly(self.essential, word_vector)
      else:
        cuboid = build_plain_cuboid(doc_num, self.essential, self.large_data, self.train_type)
    else:
      cuboid = build_plain_interpolated(self.doc_nums[idx], self.essential, self.large_data, self.train_type)
    
    
    return label, cuboid, is_augmented


def extract_word_vector(cuboid, line, keywords):
  word_vector = dict()
  cuboid = cuboid.cpu().numpy()
  for keyword, is_live, vector in zip(keywords, line, cuboid):
    if is_live > 0: word_vector[keyword] = vector
  return word_vector



def extract_label(cuboid, line, category_center, category_line, net):
  # min_dist = 1e30
  # min_label = None
  
  # flat_cuboid = cuboid.view(-1)
  # for label, center in category_center.items():
  #   flat_center = center.view(-1)
  #   # sim = torch.sum(flat_cuboid*flat_center)/(torch.norm(flat_cuboid, 2)*torch.norm(flat_center, 2))
  #   dist = torch.sum(torch.square(flat_cuboid - flat_center))
  #   if torch.isnan(dist): raise Exception()
  #   if dist < min_dist:
  #     min_dist = dist
  #     min_label = label
  
  # return min_label

  min_dist = 1e30
  min_label = None
  
  for label, line_center in category_line.items():
    # flat_center = line_center.view(-1)
    # sim = torch.sum(flat_cuboid*flat_center)/(torch.norm(flat_cuboid, 2)*torch.norm(flat_center, 2))
    dist = torch.sum(torch.square(line - line_center))
    if torch.isnan(dist): raise Exception()
    if dist < min_dist:
      min_dist = dist
      min_label = label
  
  return min_label, min_dist.item()


  # label = torch.argmax(net(cuboid.unsqueeze(0))[0]).item()
  # return label

  
