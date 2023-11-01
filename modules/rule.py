from apriori_python import apriori
from collections import defaultdict
from tqdm import tqdm
import mlxtend
import numpy as np
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import association_rules
from mlxtend.frequent_patterns import fpgrowth
import matplotlib.pyplot as plt

def mine_rules(args, essential, large_data):
    print("Rule Mining")
    method = 'fpgrowth'
    doc_label_map = large_data['train_doc_label_map']
    embedding = large_data['train_embedding']

    word_counts = defaultdict(int)
    for doc, word_vector in embedding.items():
        words = list(word_vector.keys())
        for word in words:
            word_counts[word]+=1
    words, counts = zip(*sorted(list(word_counts.items()), key=lambda x:x[1]))
    plt.plot(counts)
    plt.savefig("word_count.png")
    plt.close()
    filtered_words = set(words[:int(0.9*len(words))])

    label_keys = defaultdict(list)
    for doc, word_vector in embedding.items():
        label = doc_label_map[doc]
        words = list(word_vector.keys())
        words = [w for w in words if w in filtered_words]
        label_keys[label].append(words)
        
    label_rules = dict()
    for label, items in tqdm(label_keys.items()):
        if method == 'apriori':
            freq_itemset, rules = run_apriori(items)
            label_rules[label] = rules
        else:
            label_rules[label] = run_fpgrowth(items)
    large_data['label_rules'] = label_rules


def run_apriori(items):
    freq_itemset, rules = apriori(items, 0.5, 0.5)
    return (freq_itemset, rules)

def run_fpgrowth(items):
    te = TransactionEncoder()
    te_ary = te.fit_transform(items)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    min_support = 0.01
    min_confidence = 0.7
    result = fpgrowth(df,min_support=min_support, use_colnames=True)
    result_chart = association_rules(result, metric="confidence", min_threshold=min_confidence)
    return_result = []
    for row in result_chart.iterrows():
        return_result.append(())
    print(result_chart)
    return result_chart