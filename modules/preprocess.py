
from nltk.tokenize import word_tokenize
from nltk.stem import  PorterStemmer, WordNetLemmatizer
import nltk
from tqdm import tqdm
from sklearn.feature_extraction.text  import TfidfVectorizer
from sklearn.feature_selection import  SelectKBest, chi2
from collections import defaultdict
from transformers import AutoTokenizer
import re
import os
import pickle
from nltk.corpus import stopwords
import enchant

from embedding import merge_tokens, cleanse_word

tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-tiny', model_max_length = 512) 

tokenizer_options = {
    "padding" : 'max_length', 
    "return_token_type_ids" : True, 
    "return_attention_mask" : False,
    "truncation" : True
}


def generate_keywords(args, essential, force=True):
  data_ratio = args['data_ratio']
  dataset_name = args['dataset_name']
  train_dataset = essential['train_dataset']
  all_categories = essential['all_categories']

  print("Generating Keywords")

  os.makedirs(f"./data/{dataset_name}", exist_ok=True)
  keyword_filename = f"./data/{dataset_name}/{data_ratio}_keywords.txt"
  if os.path.exists(keyword_filename) and not force:
    with open(keyword_filename, 'r') as f:
      keywords = [w.strip() for w in f.readlines()]
  else:
    UNT = 1 #0.8
    LNT = 0 #0.001
    TARGET_NUM_KEYWORDS = args['target_num_keywords']
    
    keywords = preprocess(train_dataset, all_categories, UNT, LNT, TARGET_NUM_KEYWORDS)

    word_dictionary = enchant.Dict("en_US")
    g_correct = []
    g_wrong = []
    for word in keywords:
      if word_dictionary.check(word):
        g_correct.append(word)
      else:
        g_wrong.append(word)
    keywords = g_correct + g_wrong

    keywords = sorted(keywords[:TARGET_NUM_KEYWORDS])
    with open(keyword_filename, 'w') as f:
      for w in keywords:
        f.write(w+'\n')
  NUM_KEYWORDS = len(keywords)
  print(f"Total {NUM_KEYWORDS} keywords")
  keywords_index_map = {word:keywords.index(word) for word in keywords}
  essential['keywords_index_map'] = keywords_index_map
  essential['keywords'] = keywords
  args['num_keywords'] = NUM_KEYWORDS


def preprocess(train_dataset, all_categories, UNT, LNT, TARGET_NUM_KEYWORDS):
    result = []
    dataset = train_dataset
    for idx in tqdm(range(len(dataset))):
        row = dataset[idx]
        line = row['text']
        words = tokenize_words(line)

        clean_words = []
        for word in words:
          clean_word = cleanse_word(word)
          if len(clean_word)>0:
            clean_words.append(clean_word)
        
        # clean_words = [lemma.lemmatize(t) for t in clean_words] # lemmatizing
        # words = filter_stopwords(words)
        # words = filter_singles(words)
        if len(clean_words) > 0:
          category = all_categories[row['label']]
          result.append((category, clean_words))
    cat_document_count, word_cat_count, denoised_keywords = filter_noise_words(result, UNT, LNT, TARGET_NUM_KEYWORDS)
    keywords = chi_square(result, denoised_keywords)
    # keywords = bayesian_filter(cat_document_count, word_cat_count, denoised_keywords, all_categories)
    return keywords

def chi_square(result, denoised_keywords):
  denoised = set(denoised_keywords)
  labels, words_list = zip(*result)
  texts = [" ".join([w for w in words if w in denoised]) for words in words_list]
  vectorizer = TfidfVectorizer(max_df = 0.2)
  term_doc = vectorizer.fit_transform(texts)
  ch2 = SelectKBest(chi2, k='all')
  ch2.fit(term_doc, labels)
  scored = list(zip(vectorizer.get_feature_names_out(), ch2.scores_))
  scored.sort(key=lambda x:x[1], reverse=True)
  words = [s[0] for s in scored]
  print(f"Chi-Square Applied: {len(words)}")
  return words

def bayesian_filter(cat_document_count, word_cat_count, survived_keywords, all_categories):
  D = sum(cat_document_count.values())
  keywords = []
  for w in survived_keywords:
    S_w = sum(word_cat_count[w].values())
    for cat in all_categories:
      S_Li = word_cat_count[w][cat]/cat_document_count[cat]
      score = S_Li*D/S_w
      keywords.append((w, score))
  keywords.sort(key=lambda x:x[1], reverse=True)
  deduplicated = []
  for word, score in keywords:
    if word not in deduplicated:
      deduplicated.append(word)
  print(f"Bayesian filter: reduced to {len(deduplicated)}")
  return deduplicated
    
def filter_pos(words):
  pos_tagged = nltk.pos_tag(words)
  NN_VB = filter(lambda x:x[1].startswith("NN") or x[1].startswith("VB") or x[1].startswith("RB") or x[1].startswith("JJ"), pos_tagged)
  return [x[0] for x in NN_VB]

def filter_noise_words(doc_words, UNT, LNT, TARGET_NUM_KEYWORDS):
  doc_length = len(doc_words)
  word_cat_count = defaultdict(lambda : defaultdict(int))
  cat_document_count = defaultdict(int)
  for cat, words in tqdm(doc_words):
    cat_document_count[cat]+=1
    for word in set(words):
      word_cat_count[word][cat] += 1
  word_cat_count = dict(word_cat_count)

  keywords = []
  for word, cat_counts in word_cat_count.items():
    if LNT <= sum(cat_counts.values())/doc_length <= UNT:
      keywords.append(word)
  keyword_length = len(keywords)
  if keyword_length <= TARGET_NUM_KEYWORDS:
    LNT = 0
    keywords = []
    for word, cat_counts in word_cat_count.items():
      if LNT <= sum(cat_counts.values())/doc_length <= UNT:
        keywords.append(word)
  
  print(f"Noise word filtering(LNT={LNT}, UNT={UNT}): {len(word_cat_count.keys())} -> {len(keywords)} keywords")
  return cat_document_count, word_cat_count, keywords

def tokenize_words(sentence):
  indexed_tokens = tokenizer(sentence, **tokenizer_options)['input_ids']
  tokens = [tokenizer.convert_ids_to_tokens(ids) for ids in indexed_tokens]
  merged_tokens, _ = merge_tokens(tokens, range(len(tokens)))
  extracted_words = [t for t in merged_tokens if "[" not in t]

  return extracted_words
  # return word_tokenize(sentence)

def filter_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    filtered = [w for w in tokens if not w.lower() in stop_words]
    return filtered

def filter_singles(tokens):
    filtered = [w for w in tokens if len(w)>1]
    return filtered
