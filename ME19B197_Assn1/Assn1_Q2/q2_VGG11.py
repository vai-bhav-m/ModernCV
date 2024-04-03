# Author: Vaibhav Mahapatra (ME19B197)

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import sys
import pickle

from architectures import VGG11Classifier, MLPClassifier, VGG11Classifier_BN
from helperfuncs import model_trainer, test_performance, plot_loss_acc, heatmap_conf_matrix

# Load CIFAR-10 dataset
input_size = 32 * 32 * 3
output_size = 10
batch_size = 64
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)


# Fitting VGG11 Architecture on various hyperparameters
# n_epochs = int(sys.argv[1])
# lr = 0.005

# perfs = []
# lr_list = [0.0005, 0.001, 0.005, 0.01]
# n_hidlist = [1,5,10]
# n_unit_list = [1024, 2048, 4096, 8192]
# for lr in lr_list:
#     model = VGG11Classifier(n_hidden=1, n_units=4096).to(device)
# model = MLPClassifier(input_size, output_size, with_bn=True).to(device)

#     model, model_performance = model_trainer(model, train_loader=train_loader, val_loader=val_loader, epochs=n_epochs, learning_rate=lr)
#     train_time = model_performance["training_time"]
#     print(f"Training is complete. It took {train_time:.4f} mins to train this network for {n_epochs} epochs \n")

#     # test_acc, conf_matrix = test_performance(model, test_loader=test_loader)
#     perfs.append(model_performance)

# plot_loss_acc(perfs, labels = [f"lr={x}" for x in lr_list], caption="VGG11 w BatchNorm", filename="VGG11_w_bn")

# Final Hyperparams
bn = True 
lr = 0.01
n_hidden = 5 
n_units = 2048
n_epochs = 25

model = VGG11Classifier_BN(n_hidden=n_hidden, n_units=n_units).to(device)

model, model_performance = model_trainer(model, train_loader=train_loader, val_loader=val_loader, epochs=n_epochs, learning_rate=lr)
train_time = model_performance["training_time"]
print(f"Training is complete. It took {train_time:.4f} mins to train this network for {n_epochs} epochs \n")

test_acc, conf_matrix = test_performance(model, test_loader=test_loader)

# torch.save(model, "./best_model.pth")

# # # Save the list
# with open("./best_model_performance.pkl", 'wb') as file:
#     pickle.dump(model_performance, file)

# with open(list_path, 'rb') as file:
#     loaded_list = pickle.load(file)

# print(loaded_list)

# Generating Plots for the best model
# best_model = torch.load("./best_model.pth")
# with open("./best_model_performance.pkl", "rb") as file:
#     best_perf = pickle.load(file)


best_model = torch.load("./best_model.pth")
test_acc, conf_matrix = test_performance(best_model, test_loader=test_loader)
print(f"Test accuracy = {test_acc*100:.4f} %")
heatmap_conf_matrix(conf_matrix, test_dataset.classes, "Best Model's Confusion matrix", "best_conf")