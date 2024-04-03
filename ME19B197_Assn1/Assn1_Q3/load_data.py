# Author: Vaibhav Mahapatra (ME19B197)

import torch
from torch.utils.data import Dataset

def get_dataset(dataset_path):
    lines = open(dataset_path, encoding='utf-8').read().strip().split('\n')
    eng = [x.split("\t")[1] for x in lines]
    hin = [x.split("\t")[0] for x in lines]
    counts = [int(x.split("\t")[2]) for x in lines]    
    return eng, hin, counts

start_token = 0
end_token = 1

class Language:
  def __init__(self, name, train_words):
    self.name = name
    self.char2index = {}
    self.index2char = {0: "SOW", 1: "EOW"}      # mapping from index to character
    self.n_chars = 2                            # Count SOW, EOW 
    self.max_size = 2                           # to find the maximum length of the word we're training our model on

    for word in train_words:
      if len(word) + 2 > self.max_size:
        self.max_size = len(word) + 2

      for c in word:
        if c not in self.char2index:
          self.char2index[c] = self.n_chars
          self.index2char[self.n_chars] = c
          self.n_chars += 1
  
  def get_index(self, character):
    return self.char2index[character]
    
  def get_character(self, index):
    return self.index2char[index]
  
  def encode_word(self, word):
    coded = [self.get_index(letter) for letter in word]
    coded.append(end_token)         # Signifying EOW
    return coded
  
  def decode_word(self, word):
    if word[-1] == end_token:
        word = word[:-1]

    characters = [self.get_character(num) for num in word]
    decoded = ''.join(characters)
    return decoded

class LangDataset(Dataset):
    def __init__(self, language1, x_words, language2, y_words, device):
      self.device = device
      self.input_lang = language1
      self.inputs = [torch.tensor(language1.encode_word(x), dtype=torch.long, device=device).view(-1, 1) for x in x_words]
      self.target_lang = language2
      self.targets = [torch.tensor(language2.encode_word(y), dtype=torch.long, device=device).view(-1, 1) for y in y_words]

    def __len__(self):
      return len(self.targets)
    
    def __getitem__(self, idx):
      return self.inputs[idx], self.targets[idx]