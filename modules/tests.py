import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

from models import Classifier, criterion


def run_test(dataloader, args, essential, large_data, test_type, silent=False, model=None, compressor=None):
  num_category = len(essential['all_categories'])

  if model:
    net = model
  else:
    k = len(essential['keywords'])
    f = 128 # feature_size
    c = len(essential['all_categories'])
    net = Classifier(k ,f, c).cuda()
    net.load_state_dict(torch.load(args["path"]))
  net.eval()

  val_loss_sum = 0
  total_batches = len(dataloader)
  pred_list = []
  label_list = []
  with torch.no_grad():
    for _, batch in enumerate(dataloader):
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
  f1 = f1_score(label_list, pred_list, average='macro')

  print(f"{test_type}: avg_loss={val_loss_sum/total_batches}, accuracy={accuracy}, f1_score={f1}")
  val_loss_sum = val_loss_sum / total_batches
  return val_loss_sum, pred_list, label_list, accuracy, f1

