# coding = utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

import math

from pytorch_transformers import BertModel,BertConfig

eos_token = 100

class Encoder(nn.Module):
    def __init__(self, d_model, nhead=4, dim_feedforward=512, dropout=0.2, num_layers=1):
        super(Encoder, self).__init__()
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.encoder = torch.nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

    def forward(self, input):
        output = self.encoder(input.transpose(0, 1))
        return output.transpose(0, 1)

class Decoder(nn.Module):
    def __init__(self, d_model, nhead=4, dim_feedforward=512, dropout=0.2, num_layers=1):
        super(Decoder, self).__init__()
        decoder_layer = torch.nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.decoder = torch.nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )

    def forward(self, tgt, memory):
        output = self.decoder(tgt, memory)
        return output.transpose(0, 1)

class EncoderDecoder(nn.Module):
    def __init__(self, d_model, n_vocab, nhead=4, dim_feedforward=512, dropout=0.2, num_layers=1):
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(
            d_model = d_model, 
            nhead = nhead, 
            dim_feedforward = dim_feedforward, 
            dropout = dropout, 
            num_layers = num_layers
        )

        self.decoder = Decoder(
            d_model = d_model, 
            nhead = nhead, 
            dim_feedforward = dim_feedforward, 
            dropout = dropout, 
            num_layers = num_layers
        )

        self.tgt_word_proj = nn.Linear(d_model, n_vocab)
        nn.init.xavier_normal_(self.tgt_word_proj.weight)

    def forward(self, input_word_ids, output_word_ids):
        # Encoder
        hidden_states = self.encoder(input_word_ids)

        # Decoder
        hidden_states = self.decoder(output_word_ids, hidden_states)

        # Proj
        logits = self.tgt_word_proj(hidden_states)

        return logits

    def generate(self, input_word_ids, output_word_ids, max_generation_len):
        # Encoder
        hidden_states = self.encoder(input_word_ids)

        # Decoder
        generated_word = torch.tensor([0])
        eos_token = torch.tensor([1])
        for _ in range(max_generation_len):
            output_word_ids = torch.cat([output_word_ids, generated_word], dim = 1)
            generated_word_logits = self.tgt_word_proj(self.decoder(output_word_id, hidden_states))
            generated_word_logit = torch.softmax(generated_word_logits, dim = -1)
            generated_word = torch.multinomial(generated_word_logit, num_samples = 1)
            if generated_word == eos_token:
                break

        generated_tokens = output_word_ids[0].tolist()
        return generated_tokens