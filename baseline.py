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

    def forward(self, input_):
        output = self.encoder(input_.transpose(0, 1))
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
        return output

class EncoderDecoder(nn.Module):
    def __init__(self, d_model, n_vocab, nhead=4, dim_feedforward=512, dropout=0.2, num_layers=1):
        super(EncoderDecoder, self).__init__()
        self.n_vocab = n_vocab
        self.embedding = nn.Embedding(n_vocab, d_model)
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
        # Embedding
        hidden_representation = self.embedding(input_word_ids)
        output_hidden_representation = self.embedding(output_word_ids)

        # Encoder
        hidden_states = self.encoder(hidden_representation)

        # Decoder
        hidden_states = self.decoder(output_hidden_representation, hidden_states)

        # Proj
        logits = self.tgt_word_proj(hidden_states)

        # Loss
        loss_fct = CrossEntropyLoss()
        total_loss = loss_fct(logits.view(-1, self.n_vocab), output_word_ids.view(-1))

        return total_loss, logits

    def generate(self, input_word_ids, output_word_ids, max_generation_len):
        # Embedding
        hidden_representation = self.embedding(input_word_ids)

        # Encoder
        hidden_states = self.encoder(hidden_representation)

        # Decoder
        generated_text = []
        batch_size, seq_len = output_word_ids.shape
        output_word_ids = torch.zeros((batch_size, seq_len), dtype = torch.long).to(input_word_ids.device)
        generated_word = torch.tensor(batch_size * [101], dtype = torch.long).view(batch_size, 1).to(output_word_ids.device)
        eos_token = torch.tensor([102], dtype = torch.long).to(output_word_ids.device)
        for _ in range(max_generation_len):
            # Need change
            # Now: [PAD] * 15 + [CLS]
            output_word_ids = torch.cat([output_word_ids, generated_word], dim = 1)[:, 1:]
            output_hidden_representation = self.embedding(output_word_ids)
            generated_word_logits = self.tgt_word_proj(self.decoder(output_hidden_representation, hidden_states))
            generated_word_logit = torch.softmax(generated_word_logits, dim = -1)
            generated_word = torch.max(generated_word_logit, dim = -1)[1][0][0].view(1, 1)

            # Need change, need more elegant
            generated_text.append(generated_word.tolist()[0][0])
            if generated_word == eos_token:
                break

        return generated_text