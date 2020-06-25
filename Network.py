import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import *


class EncoderLayer(torch.nn.Module):
    def __init__(self, dim_val, dim_attn):
        super(EncoderLayer, self).__init__()
        self.attn = AttentionBlock(dim_val, dim_attn)
        self.fc1 = nn.Linear(dim_val, dim_val)
        self.norm1 = nn.LayerNorm(dim_val)
        self.norm2 = nn.LayerNorm(dim_val)
    
    def forward(self, x):
        a = self.attn(x)
        x = self.norm1(x + a)
        
        a = self.fc1(x)
        x = self.norm2(x + a)
        
        return x

class DecoderLayer(torch.nn.Module):
    def __init__(self, dim_val, dim_attn):
        super(DecoderLayer, self).__init__()
        self.attn1 = AttentionBlock(dim_val, dim_attn)
        self.attn2 = AttentionBlock(dim_val, dim_attn)
        self.fc1 = nn.Linear(dim_val, dim_val)
        self.fc2 = nn.Linear(dim_val, dim_val)
        
        self.norm1 = nn.LayerNorm(dim_val)
        self.norm2 = nn.LayerNorm(dim_val)
        self.norm3 = nn.LayerNorm(dim_val)
        
    def forward(self, x, enc):
        a = self.attn1(x)
        x = self.norm1(a + x)
        
        a = self.attn2(x, kv = enc)
        x = self.norm2(a + x)
        
        a = self.fc1(x)
        a = self.fc2(x)
        a = F.relu(x)
        x = self.norm3(x + a)
        return x

class Transformer(torch.nn.Module):
    def __init__(self, dim_val, dim_attn, input_size, input_seq_len, output_seq_len, n_decoder_layers = 1, n_encoder_layers = 1):
        super(Transformer, self).__init__()
        self.output_seq_len = output_seq_len
        
        #Initiate encoder and Decoder layers
        self.encs = []
        for i in range(n_encoder_layers):
            self.encs.append(EncoderLayer(dim_val, dim_attn))
        
        self.decs = []
        for i in range(n_decoder_layers):
            self.decs.append(DecoderLayer(dim_val, dim_attn))
        
        self.pos = PositionalEncoding(dim_val)
        
        #Dense layers for managing network inputs and outputs
        self.enc_input_fc = nn.Linear(input_size, dim_val)
        self.dec_input_fc = nn.Linear(input_size, dim_val)
        self.out_fc = nn.Linear(output_seq_len * dim_val, input_size)
    
    def forward(self, x):
        #encoder
        e = self.enc_input_fc(x)
       # e = torch.zeros(e.shape)
        #print(x.shape, e.shape)
        e = self.encs[0](self.pos(e))
        for enc in self.encs[1:]:
            e = enc(e)
        
        #decoder
        #print(x[:,-self.output_seq_len:].shape)
        d = self.decs[0](self.dec_input_fc(x[:,-self.output_seq_len:]), e)
        for dec in self.decs[1:]:
            d = dec(d, e)
            
            #print(d.shape)
        #output
        #print(d.shape)
        x = self.out_fc(d.flatten(start_dim=1))
        
        return x