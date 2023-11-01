
import torch
import os
import numpy as np
import random
import gc
import wandb
import umap
import seaborn as sns
import nltk
import parmap
import argparse
import matplotlib.pyplot as plt
import multiprocessing as mp
import seaborn as sns
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
from functools import partial
from pprint import pprint
import warnings

from load import load_dataset, generate_dataset, generate_dataloader
from preprocess import generate_keywords
from embedding import generate_embeddings
from gan import train_word_gan
from models import WordDecoder
from custom_datasets import GanDataset
from train import train
from augmentation import pre_augmentation
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch import autograd

autograd.set_detect_anomaly(True)

def prepare():
  gc.collect()
  torch.cuda.empty_cache()
  warnings.filterwarnings('ignore')
  SEED = 2000
  os.environ["TOKENIZERS_PARALLELISM"] = "true"
  os.environ['OMP_NUM_THREADS'] = "1"


  torch.manual_seed(SEED)
  random.seed(SEED)
  np.random.seed(SEED)
  nltk.download('punkt')
  nltk.download('stopwords')
  nltk.download('averaged_perceptron_tagger')
  nltk.download('wordnet')
  


def run(args):
  essential = {}
  large_data = {}

  load_dataset(args, essential)
  print("-"*100)
  generate_keywords(args, essential)
  print("-"*100)
  pre_augmentation(args, essential)
  print("-"*100)
  generate_dataset(args, essential)
  print("-"*100)
  generate_embeddings(args, essential, large_data)
  print("-"*100)

  
  if 'gan' in args['augment_method']:

    pre_dataset = essential['train_set']
    dataset = GanDataset(args, essential, large_data, pre_dataset)
    merged = torch.zeros((args['num_keywords'], 128), dtype=torch.float32)
    counts = torch.zeros((args['num_keywords']), dtype=torch.float32)
    for label, cuboid in tqdm(dataset):
      cuboid = cuboid.cpu()
      merged+=cuboid
      counts+=torch.where(torch.sum(torch.abs(cuboid), 1)> 0, 1, 0)
    counts = torch.clamp(counts, 1)
    merged /= counts.view(args['num_keywords'], 1)
    for c in counts.tolist():
      assert c>0, c
    counts /= len(dataset)

    if args['residual'] == 1:
      large_data['base_cuboid'] = merged
    else:
      large_data['base_cuboid'] = torch.zeros_like(merged)

    if args['line_augment_ratio'] > 0:

      ## Word Group
      print("Creating Word Group")
      K = args['word_group_size']
      word_groups = dict()
      keywords = essential['keywords']
      distances = []
      results = parmap.map(_generate_word_group, range(args['num_keywords']), keywords=keywords, embeddings=merged, K=K, pm_pbar=True, pm_processes=5)
      for i, (words, dists) in enumerate(results):
        distances += dists
        word_groups[keywords[i]] = words

      plt.plot(sorted(distances))
      plt.savefig("cache/distances")
      plt.close()
      
      with open('cache/word_group.txt', 'w') as f:
        for word, group in word_groups.items():
          f.write(f"{word}: {group}\n")

      large_data['word_groups'] = word_groups


    category_center = defaultdict(int)
    category_counts = defaultdict(int)
    for label, cuboid in tqdm(dataset):
      label = label.item()
      line = torch.where(torch.norm(cuboid, 2, 1)>0, 1.0, 0.0)
      category_center[label] += cuboid
      category_counts[label] += 1
    for label in category_center.keys():
      category_center[label] /= category_counts[label]


    large_data['category_center'] = category_center

    order_encoding = torch.zeros((args['num_keywords'], 128), device='cuda')
    pos = torch.arange(0, args['num_keywords'], dtype=torch.float, device='cuda').unsqueeze(dim=1)
    _2i = torch.arange(0, 128, step= 2, dtype=torch.float, device='cuda')
    order_encoding[:, 0::2] = torch.sin(pos / 10000 ** (_2i / 128))
    order_encoding[:, 1::2] = torch.cos(pos / 10000 ** (_2i / 128))
    if args["style"] == 1:
      large_data['order_encoding'] = order_encoding
    else:
      large_data['order_encoding'] = torch.zeros_like(order_encoding, device='cuda')

  if 'gan' in args['augment_method'] and args['augment_ratio'] > 0:
    train_word_gan(args, essential, large_data)
    print("-"*100)

  # print_word_plots(args, essential, large_data)
  # print_cuboid_plots(args, essential, large_data)
    
  generate_dataloader(args, essential, large_data)
  print("-"*100)
  train(args, essential, large_data)

def _generate_word_group(i, keywords, embeddings, K):
  base_word = keywords[i]
  base_emb = embeddings[i]
  dist_list = []
  for j, word in enumerate(keywords):
    other_emb = embeddings[j]
    dist = torch.mean(torch.square(base_emb-other_emb)).item()
    dist_list.append((word, dist))
  dist_list.sort(key=lambda x:x[1])
  dist_list = dist_list[:K]
  dist_list = [d for d in dist_list if d[1] <= args['max_word_distance']]
  words, dists = zip(*dist_list)
  return (words, dists)

def print_cuboid_plots(args, essential, large_data):
  num_keywords = len(essential['keywords'])
  num_features = 128 # feature size
  num_category = len(essential['all_categories'])
  word_decoder = WordDecoder(num_keywords, num_category, num_features, large_data['order_encoding'], large_data['base_cuboid'], args["word_latent"]).cuda()
  word_decoder.load_state_dict(torch.load("./word_decoder.pt"))
  train_dataset = GanDataset(args, essential, large_data, essential['train_set'], 'train')
  test_dataset = GanDataset(args, essential, large_data, essential['test_set'], 'test')
  
  types = []
  cuboids = []
  labels = []
  sample_count = 40
  indices = random.sample(list(range(len(train_dataset))), sample_count)
  for idx in tqdm(indices):
    label, cuboid = train_dataset[idx]
    line = torch.where(torch.norm(cuboid, p=2, dim=1)>0, 1, 0).view(-1, 1)
    cuboid = cuboid.view(-1).cpu().numpy()
    cuboids.append(cuboid)
    types.append(f"Original")
    labels.append(label)

    word_latent = torch.normal(0, 1, (1, args['num_keywords'], word_decoder.latent_size)).cuda()
    new_cuboid = word_decoder(word_latent, label)[0]
    new_cuboid *= line
    cuboids.append(new_cuboid.view(-1).detach().cpu().numpy())
    types.append(f"Ours")
    labels.append(label)

  
  # indices = random.sample(list(range(len(test_dataset))), sample_count)
  # count = 0
  # for idx in tqdm(indices):
  #   label, cuboid = test_dataset[idx]
  #   cuboid = cuboid.view(-1).cpu().numpy()

  #   cuboids.append(cuboid)
  #   types.append(f"Test")
  #   labels.append(label)
  #   count+=1

  
  labels = [l.item() for l in labels]

  METHOD = "umap"

  if METHOD == 'pca':
    pca = PCA(n_components=2)
    embedded = pca.fit_transform(np.array(cuboids))
  elif METHOD == 'umap':
    reducer = umap.UMAP()
    embedded = reducer.fit_transform(np.array(cuboids))

  
  df = pd.DataFrame(embedded, columns=["X", "Y"])
  df["type"] = types
  df['label'] = labels

  plt.figure(figsize=(12, 9))
  sns.scatterplot(
    x='X', 
    y='Y',
    hue='type', 
    style='label',
    s=100,
    data=df
  )
  plt.savefig(f"./data/{args['dataset_name']}_{METHOD}.png")
  plt.close()

def print_word_plots(args, essential, large_data):
  target_keywords = ["east", "eastern", "year", "years"]
  num_keywords = len(essential['keywords'])
  num_features = 128 # feature size
  num_category = len(essential['all_categories'])
  word_decoder = WordDecoder(num_keywords, num_category, num_features, large_data['order_encoding'], large_data['base_cuboid'], args["word_latent"]).cuda()
  word_decoder.load_state_dict(torch.load("./word_decoder.pt"))
  train_dataset = GanDataset(args, essential, large_data, essential['train_set'], 'train')
  test_dataset = GanDataset(args, essential, large_data, essential['test_set'], 'test')
  
  types = []
  embeddings = []
  labels = []
  words = []
  sample_count = 50
  
  for word in target_keywords:
    index = essential['keywords'].index(word)
    count = 0
    for label, cuboid in tqdm(train_dataset):
      if count >= sample_count: break
      row = cuboid.cpu().numpy()[index]
      if sum(abs(row)) == 0: continue

      embeddings.append(row)
      types.append(f"Train")
      words.append(word)
      labels.append(label)
      count+=1
    
    count = 0
    for label, cuboid in tqdm(test_dataset):
      if count >= sample_count: break
      row = cuboid.cpu().numpy()[index]
      if sum(abs(row)) == 0: continue

      embeddings.append(row)
      types.append(f"Test")
      words.append(word)
      labels.append(label)
      count+=1

    for _ in tqdm(range(sample_count)):
      word_latent = torch.normal(0, 1, (1, args['num_keywords'], word_decoder.latent_size)).cuda()
      label = random.choice(labels)
      new_cuboid = word_decoder(word_latent, label)[0]
      embeddings.append(new_cuboid[index].detach().cpu().numpy())
      types.append(f"Ours")
      words.append(word)
      labels.append(label)

      new_cuboid = word_decoder(word_latent, label)[0]
      embeddings.append(new_cuboid[index].detach().cpu().numpy())
      types.append(f"No style")
      words.append(word)
      labels.append(label)
  
  labels = [l.item() for l in labels]
  tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=15)
  embedded = tsne.fit_transform(np.array(embeddings))
  df = pd.DataFrame(embedded, columns=["PC1", "PC2"])
  df["type"] = types
  df['word'] = words
  df['label'] = labels

  plt.figure(figsize=(12, 9))
  sns.scatterplot(
    x='PC1', 
    y='PC2',
    hue='word', 
    style='type',
    s=100,
    data=df
  )
  plt.savefig(f"./data/{args['dataset_name']}_tsne.png")
  plt.close()


def print_statistics(args, essential, large_data):
  num_keywords = len(essential['keywords'])
  num_features = 128 
  num_category = len(essential['all_categories'])
  line_latent = args['line_latent']
  word_latent = args['word_latent']
  word_decoder = WordDecoder(num_keywords, num_category, num_features, large_data['base_cuboid'], word_latent).cuda()
  word_decoder.load_state_dict(torch.load("./word_decoder.pt"))
  word_decoder.eval()
  # line_decoder.eval()
  pre_dataset = essential['train_set']
  dataset = GanDataset(args, essential, large_data, pre_dataset)
  real_cuboids = []
  fake_cuboids = []
  random_cuboids = []
  # real_lines = []
  # fake_lines = []
  # random_lines = []
  keyword_range = list(range(args['num_keywords']))
  with torch.no_grad():
    for _ in tqdm(range(100)):
      i = random.randint(0, len(dataset)-1)
      label, cuboid = dataset[i]
      line = torch.where(torch.norm(cuboid, 2, 1)>0, 1, 0)
      flat_cuboid = cuboid.view(args['num_keywords']*128)

      real_cuboids.append(flat_cuboid.cpu().numpy())
      # real_lines.append(line.cpu().numpy())

      # random_dist = torch.normal(0, 1, (1, line_decoder.latent_size)).cuda()
      # fake_line = line_decoder(random_dist)[0]

      random_dist = torch.normal(0, 1, (1, args['num_keywords'], word_decoder.latent_size)).cuda()
      order_encoding = large_data['order_encoding'].view(1, args['num_keywords'], 128)
      
      fake_cuboid = word_decoder(torch.cat((random_dist, order_encoding), dim=2))
      fake_cuboid = fake_cuboid * line.unsqueeze(1)
      flat_fake_cuboid = fake_cuboid.view((args['num_keywords']*128))
      fake_cuboids.append(flat_fake_cuboid.cpu().numpy())
      # fake_lines.append(fake_line.cpu().numpy())


      line_count = random.randint(45, 70)
      random_line_idx = random.sample(keyword_range, line_count)
      random_line = np.zeros((args['num_keywords']))
      for idx in random_line_idx: random_line[idx] = 1
      # random_lines.append(random_line)
      random_cuboid = np.random.random((args['num_keywords'], 128))
      random_cuboid = random_cuboid/np.linalg.norm(random_cuboid, 2, 1, keepdims=True)*35
      random_cuboid = random_cuboid * random_line[:, np.newaxis]
      random_cuboids.append(np.reshape(random_cuboid, args['num_keywords']*128))

  tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=5)
  tsne_embedded = tsne.fit_transform(np.concatenate((real_cuboids, fake_cuboids, random_cuboids), 0))
  labels = ["real"]*len(real_cuboids) + ["fake"]*len(fake_cuboids) + ["random"]*len(random_cuboids)

  df = pd.DataFrame(tsne_embedded, columns=["x", "y"])
  df['labels'] = labels
  sns.scatterplot(data=df,x='x',y='y',hue='labels')
  plt.title("Cuboid TSNE Analysis")
  plt.savefig("./TSNE_cuboid.png")
  plt.close()

  real_sim = []
  for i1, r1 in enumerate(real_cuboids):
    for i2, r2 in enumerate(real_cuboids):
      if i1==i2: continue
      sim = np.sum(r1*r2)/(np.linalg.norm(r1, 2)*np.linalg.norm(r2, 2))
      real_sim.append(sim)

  fake_sim = []
  for fake in fake_cuboids:
    for real in real_cuboids:
      sim = np.sum(fake*real)/(np.linalg.norm(fake, 2)*np.linalg.norm(real, 2))
      fake_sim.append(sim)

  random_sim = []
  for rand in random_cuboids:
    for real in real_cuboids:
      sim = np.sum(rand*real)/(np.linalg.norm(rand, 2)*np.linalg.norm(real, 2))
      random_sim.append(sim)

  print(f"[Cuboid] Real sim: {np.mean(real_sim)}, Fake sim: {np.mean(fake_sim)}, Random sim: {np.mean(random_sim)}")
  

def main(args):
  prepare()
  wandb.init(
    project = args['project_name'],
    config = args
  )
  print(', '.join([f"{key} = {val}" for key, val in args.items()]))
  run(args)
  wandb.finish()

parser = argparse.ArgumentParser(description='Cuboid Executor')
parser.add_argument('--epochs', type=int,   default=10)
parser.add_argument('--batch_size', type=int,   default=64)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--target_num_keywords', type=int, default=5000)
parser.add_argument("--label", type=str, default='default label')
parser.add_argument("--use_cuboid", type=bool, default=True)

parser.add_argument('--project_name', type=str, default='default')
parser.add_argument('--dataset_name', type=str, required=True)
parser.add_argument('--data_ratio', type=float, default=1.0)
parser.add_argument('--augment_ratio', type=float, default = 1.0)
parser.add_argument('--augment_method', type=str, default = 'gan')
parser.add_argument("--balanced", type=float, default=1)
parser.add_argument("--soft_label", type=float, default=0)
parser.add_argument("--word_group_size", type=int, default=10)
parser.add_argument("--max_word_distance", type=float, default=10)
parser.add_argument("--line_augment_ratio", type=float, default=0)
parser.add_argument("--style", type=int, default=1)
parser.add_argument("--residual", type=int, default=1)


parser.add_argument("--word_latent", type=int, default=128)
parser.add_argument("--word_epoch", type=int, default=4)
parser.add_argument("--gan_epoch", type=int, default=3)


if __name__ == '__main__':
  os.system('clear')
  args = vars(parser.parse_args())
  main(args)