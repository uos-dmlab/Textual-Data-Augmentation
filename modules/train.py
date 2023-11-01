import pandas as pd
import numpy as np
from glob import glob
import os
import torch
import time
import matplotlib.pyplot as plt
import wandb

import json
import time
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support, ConfusionMatrixDisplay, accuracy_score
import json
import matplotlib.pyplot as plt
import torch.nn.functional as F

from models import Classifier, criterion
from tests import run_test

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def plot_statistics(results, test_results, args, essential):
  dataset_name = args['dataset_name']
  all_categories = essential['all_categories']
  num_category = len(all_categories)

  if args["augment_ratio"] == 0:
    exp_num = len(glob(f'results/{dataset_name}/plain/*'))
    new_folder_name = f"results/{dataset_name}/plain/exp{exp_num:04d}/"
  else:
    exp_num = len(glob(f'results/{dataset_name}/augmented/*'))
    new_folder_name = f"results/{dataset_name}/augmented/exp{exp_num:04d}/"
  

  wandb.run.name = f"exp{exp_num:04d}"

  os.makedirs(new_folder_name, exist_ok=True)
  print(new_folder_name)
  # if stored_model:
  #   shutil.move(stored_model, new_folder_name+"plain.pt")

  with open(new_folder_name+"parameters.json", "w") as f:
    json.dump(args, f, indent=4, cls=MyEncoder)
  epochs = range(args["epochs"])
  columns = ["epoch", "train_loss", "val_loss", "precision", "recall", "f_score", "accuracy"]
  statistics = []
  accuracy_list = []
  for epoch, row in zip(epochs, results):
    train_loss, val_loss, preds, targets = row
    precision, recall, f_score, _ = precision_recall_fscore_support(targets, preds, average='macro')
    accuracy = accuracy_score(targets, preds)
    statistics.append([epoch, train_loss, val_loss, precision, recall, f_score, accuracy])
    accuracy_list.append(accuracy)

  pd.DataFrame(statistics, columns=columns).to_csv(new_folder_name+"statistics.csv", index=False)

  plt.clf()
  x = np.argmax(accuracy_list)
  y = np.max(accuracy_list)
  plt.scatter(x, y, c='r', label=f"max accuracy={y}")
  plt.plot(epochs, accuracy_list, label="accuracy")
  plt.legend()
  plt.savefig(new_folder_name+"statistics.png")

  plt.clf()
  _, _, preds, targets = results[x]
  ConfusionMatrixDisplay.from_predictions(targets, preds, labels=list(range(num_category)), display_labels=all_categories)
  plt.savefig(new_folder_name+"confusion.png")

  test_accuracy, pred_list, label_list = test_results
  plt.clf()
  ConfusionMatrixDisplay.from_predictions(label_list, pred_list, labels=list(range(num_category)), display_labels=all_categories)
  plt.savefig(new_folder_name+"test_confusion.png")

  result = {
      "accuracy": test_accuracy,
  }
  with open(new_folder_name+'test_result.json', "w") as f:
      json.dump(result, f)

  return statistics


def train(args, essential, large_data):
  epochs = args["epochs"]
  lr = args['lr']
  augment_ratio = args['augment_ratio']
  if augment_ratio == 0: args['num_clusters'] = 0

  wandb.config.update(args, allow_val_change=True)
  if args['augment_method'] == 'kmeans': wandb.config.update(large_data['cluster_quality'])
  print("\nTrain Start")
  wandb.define_metric("epoch")
  wandb.define_metric("accuracy", summary="max", step_metric="epoch")
  wandb.define_metric("val loss", summary="min", step_metric="epoch")

  num_keywords = len(essential['keywords'])
  num_features = 128 # feature size
  num_category = len(essential['all_categories'])
  net = Classifier(num_keywords, num_category, num_features).cuda()
  optimizer = optim.Adam(net.parameters(), lr=lr)
  scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.85 ** epoch)

  best_accuracy = 0
  best_test_accuracy = 0
  test_label_list = []
  test_pred_list = []

  train_dataloader = essential['train_dataloader']
  val_dataloader = essential['val_dataloader']
  test_dataloader = essential['test_dataloader']

  results = []
  for epoch in range(epochs):
    train_loss_sum = 0
    net.train()
    total_batches = len(train_dataloader)
    for num_batch, batch in enumerate(train_dataloader):
      start = time.time()
      label, cuboids, is_augmented = batch
      optimizer.zero_grad()
      cuboids = cuboids.cuda()
      label = label.cuda()
      is_augmented = is_augmented.cuda()
      pred = net(cuboids)
      loss = criterion(pred, label, is_augmented, args)
      loss.backward()
      optimizer.step()
      train_loss_sum += loss.item()
      batch_time = int((time.time()-start)*1000)/1000
      print(f"\rTrain: Epoch {epoch}: Batch {num_batch}/{total_batches}, avg loss={train_loss_sum/(num_batch+1)}, current loss={round(loss.item(), 5)}, {batch_time}s/batch"+(" "*5), end='')
    del label, cuboids, is_augmented, batch, loss
    print("")
    scheduler.step()
    train_loss_sum = train_loss_sum/total_batches
    
    val_loss_sum = 0
    total_batches = len(val_dataloader)
    pred_list = []
    label_list = []

    net.eval()
    with torch.no_grad():
      for num_batch, batch in enumerate(val_dataloader):
        label, cuboids, is_augmented = batch
        cuboids = cuboids.cuda()
        label = label.cuda()
        is_augmented = is_augmented.cuda()
        label_list += label.tolist()
        pred = net(cuboids)
        loss = criterion(pred, label, is_augmented, args)
        pred_labels = torch.max(pred, axis=1).indices
        pred_list+=pred_labels.tolist()
        val_loss_sum += loss.item()
      del label, cuboids, is_augmented, batch, loss
    accuracy = accuracy_score(label_list, pred_list)
    val_loss_sum = val_loss_sum / total_batches
    print(f"Val: Epoch {epoch}, avg_loss={val_loss_sum}, accuracy={accuracy}")
    wandb.log({"epoch": epoch, "val loss": val_loss_sum, "accuracy": accuracy})
    
    results.append((train_loss_sum, val_loss_sum, pred_list, label_list))
    if accuracy > best_accuracy:
      _, pred_list, test_label_list, test_accuracy, test_f1_score = run_test(test_dataloader, args, essential, large_data, test_type='test', model=net, compressor=None, silent=True)
      best_accuracy = accuracy
      test_pred_list = pred_list
      best_test_accuracy = test_accuracy

      wandb.run.summary.update({"test_accuracy": best_test_accuracy, "test_f1_score" : test_f1_score})

  test_results = (best_test_accuracy, test_pred_list, test_label_list)
  statistics = None # plot_statistics(results, test_results, args, essential)

  return statistics



