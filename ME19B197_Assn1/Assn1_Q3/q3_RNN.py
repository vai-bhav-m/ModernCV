# Author: Vaibhav Mahapatra (ME19B197)

import torch
from load_data import get_dataset, Language, LangDataset
from architectures import Seq2Seq, Seq2Seq_v2
import random
import time
import sys

root_path = "./data/hi.translit.sampled."
eng_train, hin_train, counts_train = get_dataset(root_path+"train.tsv")
eng_dev, hin_dev, counts_dev = get_dataset(root_path+"dev.tsv")
eng_test, hin_test, counts_test = get_dataset(root_path+"test.tsv")

English = Language('eng', eng_train)
Hindi = Language('hindi', hin_train)

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

train_dataset = LangDataset(English, eng_train, Hindi, hin_train, device=device)
val_dataset = LangDataset(English, eng_dev, Hindi, hin_dev, device=device)
test_dataset = LangDataset(English, eng_test, Hindi, hin_test, device=device)


# Hyperparameters
embedding_size = 128
hidden_size = 256
dropout=0
lr=0.01

# model = Seq2Seq(inp_language = English, out_language = Hindi, embedding_size = embedding_size, n_layers = 4, hidden_size=hidden_size, device=device, dropout=0, lr=0.0001)
model = Seq2Seq_v2(inp_language = English, out_language = Hindi, embedding_size = embedding_size, n_layers = 4, hidden_size=hidden_size, device=device, dropout=0, lr=0.0001)

n_iters = 200000
training_pairs = [random.choice(train_dataset) for i in range(n_iters)]

train_loss, train_c = 0, 0
losses, train_accs, test_accs = [], [], [[], []]
start_time = time.time()
for i, pair in enumerate(training_pairs):
  x = pair[0]
  y = pair[1]
  loss, match = model.train_step(x, y)
  train_c += match
  train_loss = train_loss + loss

  if (i+1)%5000 == 0:
    print('------------------------------------------------')
    print('Train loss is:', train_loss/5000)
    test_acc, test_char_acc = model.evaluate(val_dataset)
    
    losses.append(train_loss/5000)
    
    train_accs.append(train_c/5000)
    test_accs[0].append(test_acc)
    test_accs[1].append(test_char_acc)

    print(f'Validation acc is {test_acc:.5f} and character-wise accuracy is {test_char_acc:.5f}')
    print(f"{(i+1)/len(training_pairs)*100:.4f} % iterations done in {(time.time()-start_time)/60:.3f} mins")
    train_loss, train_c = 0, 0

caption = "v2"
torch.save(model, caption+".pth")
print("Model saved!")


import pickle
filename = "model_perf" + caption # sys.argv[1]
with open(filename+".pkl", 'wb') as file:
    pickle.dump([losses, train_accs, test_accs], file)
print("Model Performance saved!")
