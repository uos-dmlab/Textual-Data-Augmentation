import torch
import random
import torch.nn.functional as F

from augmentation import augment_keywords


def build_cuboid(args, essential, large_data, doc_nums, kmeans_data, augment_ratio=0, train_type="train"): # batched tokens, features
    embedding = large_data[train_type+'_embedding']
    keywords = essential['keywords']
    keywords_index_map = essential['keywords_index_map']
    feature_size = 128
    cuboids = torch.zeros(len(doc_nums), len(keywords), feature_size, dtype=torch.float32, device='cuda')
    is_augmented = []
    # doc_nums = doc_nums.tolist()
    for i, doc_num in enumerate(doc_nums):
      word_vector = embedding[doc_num]
      if augment_ratio > 0 and random.random() < augment_ratio: # augment cuboid
        change_type = args['change_type']
        change_ratio = args["change_ratio"]
        if change_type == "random":
          change_type = random.choice(["exchange", "insert", "delete"])
        new_keyword_features = augment_keywords(args, doc_num, word_vector, kmeans_data, change_ratio, train_type, change_type)
        for keyword, feature in new_keyword_features.items():
          row = keywords_index_map[keyword] 
          cuboids[i][row] = torch.from_numpy(feature)
        is_augmented.append(True)
      else: # not augmented
        for keyword, feature in word_vector.items():
          row = keywords_index_map[keyword]
          cuboids[i][row] = torch.from_numpy(feature)
        is_augmented.append(False)
    return cuboids, is_augmented

def build_plain_cuboid(doc_num, essential, large_data, train_type):
  embedding = large_data[train_type+'_embedding']
  keywords = essential['keywords']
  keywords_index_map = essential['keywords_index_map']
  feature_size = 128
  cuboid = torch.zeros(len(keywords), feature_size, dtype=torch.float32, device='cuda')
  word_vector = embedding[doc_num]
  for keyword, feature in word_vector.items():
    row = keywords_index_map[keyword]
    cuboid[row] = torch.from_numpy(feature)
  return cuboid

def build_plain_cuboid_directly(essential, word_vector):
  feature_size = 128
  keywords = essential['keywords']
  keywords_index_map = essential['keywords_index_map']
  cuboid = torch.zeros(len(keywords), feature_size, dtype=torch.float32, device='cuda')
  for keyword, feature in word_vector.items():
    row = keywords_index_map[keyword]
    cuboid[row] = torch.from_numpy(feature)
  return cuboid

def build_plain_interpolated(doc_num, essential, large_data, train_type):
  embedding = large_data[train_type+'_embedding']
  feature_size = 128
  features = list(embedding[doc_num].values())
  if len(features) == 0:
    keywords_tensor = torch.zeros(feature_size, dtype=torch.float32, device='cuda').unsqueeze(0)
  else:
    keywords_tensor = torch.tensor(features, device='cuda')
  keywords_tensor = keywords_tensor.unsqueeze(0).unsqueeze(0)
  assert len(keywords_tensor.shape) == 4, f"{keywords_tensor.shape}, {len(features)}"
  cuboid = F.interpolate(keywords_tensor, size=(len(essential['keywords']), feature_size))[0][0]
  return cuboid