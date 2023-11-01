import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import random
from sklearn.datasets import fetch_20newsgroups
from datasets import load_dataset as load_huggingface_dataset
import os
from torch.utils.data import DataLoader
import math
import matplotlib.pyplot as plt

from custom_datasets import PreDataset, CuboidDataset


def create_dataset_list(label, indices, texts):
    dataset = []
    for idx in indices:
        dataset.append({
            "label": label,
            "text": texts[idx],
            "augmented" : False
        })
    return dataset

def single_category(topics, all_categories):
    included = [t for t in topics if t in all_categories]
    if len(included) == 1: return included[0]
    return False



def load_dataset(args, essential):
    dataset_name = args['dataset_name']
    data_ratio = args['data_ratio']
    balanced = args['balanced']

    if dataset_name == 'reuters':
        dataset = load_huggingface_dataset("reuters21578", "ModApte")
        all_categories = [
            'earn', 
            'acq', 
            'crude', 
            'trade', 
            'money-fx', 
            'interest', 
            'money-supply', 
            'ship', 
            'sugar', 
            'coffee'
        ]
        
        train = dataset['train']
        test = dataset['test']

        topics = train['topics'] + test['topics']
        titles = train['title'] + test['title']
        texts = train['text'] + test['text']
        texts = [title+" "+text for title, text in zip(titles, texts)]


        filtered_labels = []
        filtered_texts = []
        for topic, text in zip(topics, texts):
            cat = single_category(topic, all_categories)
            if cat:
                filtered_labels.append(all_categories.index(cat))
                filtered_texts.append(text)

        labels = filtered_labels
        texts = filtered_texts


    elif dataset_name == 'newsgroup':
        train = fetch_20newsgroups(data_home='./cache', subset='train')
        test = fetch_20newsgroups(data_home='./cache', subset='test')

        all_categories = list(train.target_names)
        raw_labels = np.concatenate((train.target, test.target))
        raw_texts = train.data + test.data


        selected_categories = all_categories # ['comp.graphics', 'sci.med', 'rec.sport.baseball', 'talk.religion.misc']
        selected_labels = [all_categories.index(cat) for cat in selected_categories]
        for label in selected_labels: assert label > -1

        labels = []
        texts = []

        for label, text in zip(raw_labels, raw_texts):
            if label in selected_labels:
                assert selected_labels.index(label) > -1
                labels.append(selected_labels.index(label))
                texts.append(text)
        all_categories = selected_categories
        
    
    elif dataset_name == 'twitter':
        df = pd.read_csv("data/twitter.csv")
        df = df[df.airline_sentiment_confidence >= 0.6]

        
        categories = df.airline_sentiment.tolist()
        all_categories = ["positive", "negative", "neutral"]
        labels = [all_categories.index(cat) for cat in categories]
        texts = df.text.tolist()


    elif dataset_name == 'yahoo':
        train_df = pd.read_csv("data/yahoo_train.csv").fillna("")
        test_df = pd.read_csv("data/yahoo_test.csv").fillna("")
        df = pd.concat([train_df, test_df], ignore_index=True)

        all_categories = [
            "Society & Culture",
            "Science & Mathematics",
            "Health",
            "Education & Reference",
            "Computers & Internet",
            "Sports",
            "Business & Finance",
            "Entertainment & Music",
            "Family & Relationships",
            "Politics & Government"
        ]

        labels = [label-1 for label in df['class'].astype(int)]
        texts = df['question title'] + " " + df['question content'] + " " + df['best answer']

    elif dataset_name == 'imdb':
        df = pd.read_csv("data/imdb.csv")
        all_categories = ["neg", "pos"]

        categories = df.category.tolist()
        labels = [all_categories.index(cat) for cat in categories]
        texts = df.text.tolist()
        

    elif dataset_name == 'sst2':
        dataset = load_huggingface_dataset("sst2")
        all_categories = [
            'negative',
            "positive"
        ]

        train = dataset['train']
        test = dataset['validation']
        labels = train['label'] + test['label']
        texts = train['sentence'] + test['sentence']

    else:
        raise Exception("No Dataset Name Found")
    
    train_dataset = []
    val_dataset = []
    test_dataset = []
    
    category_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        category_indices[label].append(idx)
    category_indices = dict(category_indices)

    if dataset_name == 'yahoo':
        for label in range(len(all_categories)): ## 1% Sampling
            category_indices[label] = random.sample(category_indices[label], len(category_indices[label])//100)

    if balanced == 0:
        for label in range(len(all_categories)):
            min_idx = len(category_indices[label])
            train_cut = int((min_idx//2)*0.8)
            val_cut = min_idx//2
            random.shuffle(category_indices[label])
            train_indices = category_indices[label][:train_cut]
            if data_ratio > 1:
                sample_count = min(int(data_ratio), len(train_indices))
            else:
                sample_count = math.ceil(len(train_indices)*data_ratio)
            train_indices = train_indices[:sample_count]
            val_indices = category_indices[label][train_cut:val_cut]
            test_indices = category_indices[label][val_cut:min_idx]
            train_dataset += create_dataset_list(label, train_indices, texts)
            val_dataset += create_dataset_list(label, val_indices, texts)
            test_dataset += create_dataset_list(label, test_indices, texts)
    else:
        min_idx = min([len(indices) for indices in category_indices.values()])
        train_cut = int((min_idx//2)*0.8)
        val_cut = min_idx//2
        for label in range(len(all_categories)):
            random.shuffle(category_indices[label])
            train_indices = category_indices[label][:train_cut]
            if data_ratio > 1:
                sample_count = min(int(data_ratio), len(train_indices))
            else:
                sample_count = math.ceil(len(train_indices)*data_ratio)
            train_indices = train_indices[:sample_count]
            val_indices = category_indices[label][train_cut:val_cut]
            test_indices = category_indices[label][val_cut:min_idx]
            train_dataset += create_dataset_list(label, train_indices, texts)
            val_dataset += create_dataset_list(label, val_indices, texts)
            test_dataset += create_dataset_list(label, test_indices, texts)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    essential['all_categories'] = all_categories
    essential['train_dataset'] = train_dataset
    essential['val_dataset'] = val_dataset
    essential['test_dataset'] = test_dataset


def generate_dataset(args, essential):
  train_dataset = essential['train_dataset']
  val_dataset = essential['val_dataset']
  test_dataset = essential['test_dataset']
  all_categories = essential['all_categories']
  keywords = essential['keywords']

  print("Generating Pre dataset")
  train_set = PreDataset(train_dataset, all_categories, keywords, features=True)
  val_set = PreDataset(val_dataset, all_categories, keywords,features=True)
  test_set = PreDataset(test_dataset, all_categories, keywords, features=True)

  
  essential['train_set'] = train_set
  essential['val_set'] = val_set
  essential['test_set'] = test_set

  


def generate_dataloader(args, essential, large_data):
  batch_size = args['batch_size']

  train_dataloader = DataLoader(CuboidDataset(args, essential, large_data, "train", essential['train_set']), batch_size = batch_size, num_workers=0, shuffle=True)
  val_dataloader = DataLoader(CuboidDataset(args, essential, large_data, "val", essential['val_set']), batch_size = batch_size, num_workers=0, shuffle=True)
  test_dataloader = DataLoader(CuboidDataset(args, essential, large_data, "test", essential['test_set']), batch_size = batch_size, num_workers=0, shuffle=True)

  essential['train_dataloader'] = train_dataloader
  essential['val_dataloader'] = val_dataloader
  essential['test_dataloader'] = test_dataloader