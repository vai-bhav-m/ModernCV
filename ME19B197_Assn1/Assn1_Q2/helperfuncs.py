# Author: Vaibhav Mahapatra (ME19B197)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

def batch_predict(model, inputs, labels):
  with torch.no_grad():
    inputs, labels = inputs.to(device), labels.to(device)

    # Forward pass
    outputs = model(inputs)
    loss = nn.CrossEntropyLoss()(outputs, labels)
    data_loss = loss.item()

    # Class predictions and accuracy
    y_hat = torch.argmax(outputs, 1)
    data_acc = (labels == y_hat).sum().detach().item() / len(labels)

    return y_hat, data_loss, data_acc

def model_trainer(model, train_loader, val_loader, epochs, learning_rate):
  
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(model.parameters(), lr=learning_rate)

  import time
  start_time = time.time()

  train_err, val_err, acc_train, acc_val = [], [], [], []
  for epoch in range(epochs):
      train_loss, val_loss, train_acc, val_acc = 0, 0, 0, 0

      model.eval()
      with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            y_hat, v_l, v_a = batch_predict(model, inputs, labels)
            val_loss += v_l
            val_acc += v_a 
      model.train()

      for inputs, labels in train_loader:
          inputs, labels = inputs.to(device), labels.to(device)

          outputs = model(inputs)
          loss = criterion(outputs, labels)
          train_loss += loss.item()

          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

          y_hat = torch.argmax(outputs, 1)
          train_acc += (labels == y_hat).sum().detach().item() / len(labels)


      train_err.append(train_loss/len(train_loader))
      val_err.append(val_loss/len(val_loader))
      acc_train.append(train_acc/len(train_loader))
      acc_val.append(val_acc/len(val_loader))

      if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_err[-1]:.4f}, Val Loss: {val_err[-1]:.4f}, Time elapsed: {(time.time()-start_time) / 60:.4f} mins")

  model_performance = {
      "train_loss_history": train_err,
      "val_loss_history": val_err,
      "train_acc_history": acc_train,
      "val_acc_history": acc_val,
      "training_time": (time.time()-start_time) / 60
  }

  return model, model_performance

def test_performance(model, test_loader):
    conf_matrix = np.zeros((10,10), dtype=np.int64)
    with torch.no_grad():
        for inputs, labels in test_loader:
            y_hat, _, test_acc = batch_predict(model, inputs, labels)
            for i, x in enumerate(labels):
                conf_matrix[x, y_hat[i]] += 1
    
    return test_acc, conf_matrix

def plot_loss_acc(models_perf, labels, caption, filename):
  fig, ax1 = plt.subplots(1, 1, figsize=(7, 7))
  ax1.set_xlabel('Epochs')
  ax1.set_ylabel('Accuracy')
  
  cols = ['r', 'b', 'g', 'k']
  for i, perf in enumerate(models_perf): 
    train_acc = perf["train_acc_history"]
    val_acc = perf["val_acc_history"]

    n_epochs = len(train_acc)
    
    ax1.plot(range(1, n_epochs + 1), train_acc, f"{cols[i]}-", label=f"{labels[i]} (train)")
    ax1.plot(range(1, n_epochs + 1), val_acc, f"{cols[i]}-.", label=f"{labels[i]} (val)")

  ax1.legend()

  ax1.set_title(caption)
  plt.tight_layout()
  plt.show()
  plt.savefig(filename + ".png")
  plt.close()
          
def heatmap_conf_matrix(conf, classes, caption, filename):
  plt.figure(figsize=(10, 10))
  sns.heatmap(conf, annot=True, fmt=".2f", cmap="Blues", xticklabels=classes, yticklabels=classes)
  plt.title(caption)
  plt.xlabel('Predicted Label')
  plt.ylabel('True Label')
  plt.show()
  plt.savefig(filename + ".png")
  plt.close()