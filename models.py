import torch.nn as nn
import torch
import torch.nn.functional as F

class RNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, dropout, hidden_dim, output_dim):
        
        super().__init__()
        self.d = dropout 
        
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        
        self.rnn = nn.RNN(embedding_dim, hidden_dim)
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):

        embedded = self.embedding(text)
        
        output, hidden = self.rnn(embedded)
        
        if self.d > 0 :
            hidden = self.dropout(hidden[-1,:,:])
            return self.fc(hidden)
        else : 
            return self.fc(hidden.squeeze(0))
    
    
    
class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, 
                 dropout, pad_idx):
        
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.convs = nn.Conv2d(in_channels = 1, out_channels = n_filters, 
                                                kernel_size = (filter_sizes, embedding_dim)) 
                                    
        self.fc = nn.Linear(n_filters, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        text = text.permute(1, 0)
        embedded = self.embedding(text)
        embedded = embedded.unsqueeze(1)
        conved = F.relu(self.convs(embedded)).squeeze(3)
        pooled = F.max_pool1d(conved, conved.shape[2]).squeeze(2)
        cat = self.dropout(pooled)
        return self.fc(cat)
    
 