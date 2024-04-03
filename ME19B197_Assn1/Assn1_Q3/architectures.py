# Author: Vaibhav Mahapatra (ME19B197)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import Levenshtein


class Encoder(nn.Module):
  def __init__(self, inp_vocab_size, embedding_size, n_layers, hidden_size, dropout, device):
    super(Encoder, self).__init__()
    self.vocab_size = inp_vocab_size
    self.embedding_size = embedding_size
    self.n_layers = n_layers
    self.hidden_size = hidden_size
    self.device = device

    self.embedding_layer = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_size).to(self.device)
    self.rnn = nn.RNN(input_size=self.embedding_size, hidden_size=self.hidden_size, num_layers = self.n_layers, dropout = dropout).to(self.device)
    
  def forward(self, input, hidden):
    embedded = self.embedding_layer(input).view(1, 1, -1)
    output, hidden = self.rnn(embedded, hidden)    
    return output, hidden
    
  def init_hidden(self):
    return torch.zeros(self.n_layers, 1, self.hidden_size, device = self.device)
    
class Decoder(nn.Module):
  def __init__(self, out_vocab_size, embedding_size, n_layers, hidden_size, dropout, device):
    super(Decoder, self).__init__()
    self.vocab_size = out_vocab_size
    self.embedding_size = embedding_size
    self.n_layers = n_layers
    self.hidden_size = hidden_size
    self.softmax = nn.LogSoftmax(dim=1)
    self.dropout = dropout
    self.device = device

    self.linear = nn.Linear(self.hidden_size, self.vocab_size).to(self.device)
    self.embedding_layer = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_size).to(self.device)
    self.rnn = nn.RNN(input_size=self.embedding_size, hidden_size=self.hidden_size, num_layers = self.n_layers, dropout = self.dropout).to(self.device)
  
  def forward(self, input, hidden):
    embedded = self.embedding_layer(input).view(1, 1, -1)
    output = F.tanh(embedded)
    output, hidden = self.rnn(output, hidden)
    output = self.linear(output[0])
    output = self.softmax(output)
    return output, hidden

class Seq2Seq:
  def __init__(self, inp_language, out_language, embedding_size, n_layers, hidden_size, device, dropout = 0.2, lr = 0.01):
    self.device = device
    self.encoder = Encoder(inp_language.n_chars, embedding_size, n_layers, hidden_size, dropout=dropout, device=self.device)
    self.decoder = Decoder(out_language.n_chars, embedding_size, n_layers, hidden_size, dropout=dropout, device = self.device)
    self.lr = lr
    self.max_length = out_language.max_size
    self.inp_language = inp_language
    self.out_language = out_language

    self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr = self.lr)
    self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr = self.lr)

    self.loss_fn = nn.NLLLoss()

  def train_step(self, input, target):
    encoder_hidden = self.encoder.init_hidden()

    self.encoder_optimizer.zero_grad()
    self.decoder_optimizer.zero_grad()

    input_length = input.size(0)
    for i in range(0, input_length):
      encoder_output, encoder_hidden = self.encoder(input[i], encoder_hidden)
    
    decoder_input = torch.tensor([[0]], device=self.device)  
    decoder_hidden = encoder_hidden
    
    target_length = target.size(0)
    loss = 0

    preds = []
    target_w = []
    for j in range(0, target_length):
      decoder_output, decoder_hidden = self.decoder.forward(decoder_input, decoder_hidden)
      loss = loss + self.loss_fn(decoder_output, target[j])
      value, index = decoder_output.topk(1)
      decoder_input = index.squeeze().detach()
      target_w.append(target[j].item())
      preds.append(decoder_input.item())
      if decoder_input.item() == 1:
        break
    
    loss.backward()
    self.encoder_optimizer.step()
    self.decoder_optimizer.step()

    pred_word = self.out_language.decode_word(preds)
    tar_word =  self.out_language.decode_word(target_w)
    
    return loss.item()/target_length, pred_word==tar_word
  
  def predict(self, input):
    with torch.no_grad():
      encoder_hidden = self.encoder.init_hidden()

      input_length = input.size(0)
      for i in range(0, input_length):
        encoder_output, encoder_hidden = self.encoder.forward(input[i], encoder_hidden)

      decoder_input = torch.tensor([[0]], device=self.device)
      decoder_hidden = encoder_hidden

      outputs = []
      for i in range(0, self.max_length):
        decoder_output, decoder_hidden = self.decoder.forward(decoder_input, decoder_hidden)
      
        value, index = decoder_output.data.topk(1)
        decoder_input = index.squeeze().detach()
        outputs.append(decoder_input.item())
        if decoder_input.item() == 1:
            break

      return outputs
  

  def evaluate(self, data):
    correct = 0
    character_wise = 0
    count = 0
    total_distance = 0

    for input, target in data:
      pred = self.predict(input)
      target = target.tolist()
      target = [t[0] for t in target]

      count += 1
      pred_word = self.out_language.decode_word(pred)
      tar_word =  self.out_language.decode_word(target)
      
      if pred_word == tar_word:
        correct = correct + 1
      
      total_distance = total_distance + min((Levenshtein.distance(pred_word, tar_word)/max(len(tar_word),len(pred_word))), 1)
    
    avg_distance = total_distance/len(data)
    char_acc = 1 - avg_distance
    acc = correct/len(data)

    return acc, char_acc

class Seq2Seq_v2:
  def __init__(self, inp_language, out_language, embedding_size, n_layers, hidden_size, device, dropout = 0.2, lr = 0.01):
    self.device = device
    self.encoder = Encoder(inp_language.n_chars, embedding_size, n_layers, hidden_size, dropout=dropout, device=self.device)
    self.decoder = Decoder(out_language.n_chars, embedding_size, n_layers, hidden_size, dropout=dropout, device = self.device)
    self.lr = lr
    self.max_length = out_language.max_size
    self.inp_language = inp_language
    self.out_language = out_language

    self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr = self.lr)
    self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr = self.lr)

    self.loss_fn = nn.NLLLoss()

  def train_step(self, input, target):
    encoder_hidden = self.encoder.init_hidden()

    self.encoder_optimizer.zero_grad()
    self.decoder_optimizer.zero_grad()

    input_length = input.size(0)
    for i in range(0, input_length):
      encoder_output, encoder_hidden = self.encoder(input[i], encoder_hidden)
    
    decoder_input = torch.tensor([[0]], device=self.device)  
    decoder_hidden = encoder_hidden
    hT = encoder_hidden
    
    target_length = target.size(0)
    loss = 0

    preds = []
    target_w = []
    for j in range(0, target_length):
      decoder_output, decoder_hidden = self.decoder.forward(decoder_input, decoder_hidden + hT)
      loss = loss + self.loss_fn(decoder_output, target[j])
      value, index = decoder_output.topk(1)
      decoder_input = index.squeeze().detach()
      target_w.append(target[j].item())
      preds.append(decoder_input.item())
      if decoder_input.item() == 1:
        break
    
    loss.backward()
    self.encoder_optimizer.step()
    self.decoder_optimizer.step()

    pred_word = self.out_language.decode_word(preds)
    tar_word =  self.out_language.decode_word(target_w)
    
    return loss.item()/target_length, pred_word==tar_word
  
  def predict(self, input):
    with torch.no_grad():
      encoder_hidden = self.encoder.init_hidden()

      input_length = input.size(0)
      for i in range(0, input_length):
        encoder_output, encoder_hidden = self.encoder.forward(input[i], encoder_hidden)

      decoder_input = torch.tensor([[0]], device=self.device)
      decoder_hidden = encoder_hidden
      hT = encoder_hidden

      outputs = []
      for i in range(0, self.max_length):
        decoder_output, decoder_hidden = self.decoder.forward(decoder_input, decoder_hidden + hT)
      
        value, index = decoder_output.data.topk(1)
        decoder_input = index.squeeze().detach()
        outputs.append(decoder_input.item())
        if decoder_input.item() == 1:
            break

      return outputs
  

  def evaluate(self, data):
    correct = 0
    character_wise = 0
    count = 0
    total_distance = 0

    for input, target in data:
      pred = self.predict(input)
      target = target.tolist()
      target = [t[0] for t in target]

      count += 1
      pred_word = self.out_language.decode_word(pred)
      tar_word =  self.out_language.decode_word(target)
      
      if pred_word == tar_word:
        correct = correct + 1
      
      total_distance = total_distance + min((Levenshtein.distance(pred_word, tar_word)/max(len(tar_word),len(pred_word))), 1)
    
    avg_distance = total_distance/len(data)
    char_acc = 1 - avg_distance
    acc = correct/len(data)

    return acc, char_acc
