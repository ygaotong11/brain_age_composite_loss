import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TALSTM(nn.Module):

    '''
    TA-LSTM
    This model is used for ICN training

    '''

    def __init__(self, seqlen = 122, dim = 53, hidden_size = 64,num_layer = 3, Two_linear = False):
        super(TALSTM,self).__init__()
        self.input_size = dim 
        self.hidden_dim = hidden_size
        self.num_layers = num_layer
        drp = 0.5

        self.lstm = nn.LSTM(self.input_size,self.hidden_dim,self.num_layers,bidirectional=False,batch_first=True)
        self.dropout = nn.Dropout(drp)
        self.gradients = None
        self.linear_bool = Two_linear

        if self.linear_bool == True:
            # TA-LSTM V2 (modified w/ adding additional Linear layer)
            self.relu = nn.ReLU()
            self.linear = nn.Linear(seqlen,hidden_size)
            self.out1 = nn.Linear(hidden_size,1)
        else:
            # TA-LSTM v1
            self.out = nn.Linear(seqlen,1)

        
        
    def activations_hook(self, grad):
        self.gradients = grad
        
    def attention_net(self,x,query,mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, x.transpose(1, 2)) / math.sqrt(d_k)  

        alpha_n = F.softmax(scores, dim=-1)  

        context = torch.matmul(alpha_n, x).sum(2)
        return context,alpha_n
    def forward(self, x):
        
        r_out,(h_n,h_c) = self.lstm(x,None)
        query = self.dropout(r_out)
        attn_out, alpha_n = self.attention_net(r_out,query)

        if self.linear_bool == True:
            # TA-LSTM V2  (modified) 
            out = self.relu(self.linear(attn_out))
            out = self.out1(out)
        else:
            # TA-LSTM original
            out = self.out(attn_out)

        return out
