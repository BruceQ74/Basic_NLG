# coding = utf-8

import torch

from pytorch_transformers import BertModel, BertTokenizer

model = BertModel.from_pretrained("../../../model/bert_base_uncased")
model = model.to("cuda")
tokenizer = BertTokenizer.from_pretrained("../../../model/bert_base_uncased")

token = [[101, 5290,  102,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0]]
token = torch.tensor(token, dtype=torch.long).to("cuda")

print(token)
print(token.shape)
print(token.device)
print(model.device)

ans, _ = model(token)

print(ans)


