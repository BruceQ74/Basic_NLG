# coding = utf-8

import os
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset, Dataset)

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, _input, _output = None):
        """Constructs a InputExample.
        Args:
            Input: Prompt
            Output: Generated text
        """
        self.guid = guid
        self.input = _input
        self.output = _output

class DiaDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""
    def __init__(self, data_dir, dataset):
        self.data_dir = data_dir
        self.dataset = dataset

    def get_train_examples(self):
        """See base class."""
        return self._create_examples(
            self._read_txt(os.path.join(self.data_dir, "train.txt")), "train")

    def get_dev_examples(self):
        """See base class."""
        return self._create_examples(
            self._read_txt(os.path.join(self.data_dir, "valid.txt")), "dev")

    def get_test_examples(self):
        """See base class."""
        return self._create_examples(
            self._read_txt(os.path.join(self.data_dir, "test.txt")), "test")

    def _create_examples(self, lines, set_type):
        examples = []
        input_ = lines[0][0]
        output_ = lines[0][1]
        length = len(input_)
        for i in range(length):
            guid = "%s-%s" % (set_type, i+1)
            _input = input_[i]
            _output = output_[i]
            examples.append(InputExample(guid=guid, _input=_input, _output=_output))
        return examples

    @classmethod
    def _read_txt(cls, input_file):
        '''
        read file
        return format :
        '''
        if os.path.exists(input_file) is False:
            return []
        data = []
        input_text = []
        output_text = []
        
        with open(input_file, "r", encoding = "utf-8") as f:
            for line in f:
                if len(line) == 0:
                    continue
                splits = line.strip().split('\t')
                
                input_text.append(splits[0])
                output_text.append(splits[1])

            if len(input_text) > 0:
                data.append((input_text, output_text))
        return data

    def convert_to_feature(self, tokenizer, examples, max_seq_length=16):

        # entities
        features = []
        for ex_index, example in enumerate(examples):

            labels = []
            valid_ids1 = []
            valid_ids2 = []

            input_text = example.input
            output_text = example.output

            token1 = tokenizer.tokenize(input_text)
            token2 = tokenizer.tokenize(output_text)
        
            token1 = token1[:max_seq_length - 2]
            token2 = token2[:max_seq_length - 2]
            
            for m in range(len(token1)):
                if m == 0:
                    valid_ids1.append(1)
                else:
                    valid_ids1.append(0)
            for m in range(len(token2)):
                if m == 0:
                    valid_ids2.append(1)
                else:
                    valid_ids2.append(0)

            tokens1 = ["[CLS]"] + token1 + ["[SEP]"]
            tokens2 = ["[CLS]"] + token2 + ["[SEP]"]
            valid_ids1 = [1] + valid_ids1 + [1]
            valid_ids2 = [1] + valid_ids2 + [1]
            input_ids1 = tokenizer.convert_tokens_to_ids(tokens1)
            input_ids2 = tokenizer.convert_tokens_to_ids(tokens2)

            segment_ids = [0] * max_seq_length

            if len(input_ids1) < max_seq_length:
                input_ids1 += [0] * (max_seq_length - len(input_ids1))
                valid_ids1 += [0] * (max_seq_length - len(valid_ids1))

            if len(input_ids2) < max_seq_length:
                input_ids2 += [0] * (max_seq_length - len(input_ids2))
                valid_ids2 += [0] * (max_seq_length - len(valid_ids2))

            assert len(input_ids1) == max_seq_length
            assert len(input_ids2) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(valid_ids1) == max_seq_length
            assert len(valid_ids2) == max_seq_length
            

            features.append({
                "input_ids": torch.tensor(input_ids1, dtype=torch.long),
                "output_ids": torch.tensor(input_ids2, dtype=torch.long),
                "segment_ids": torch.tensor(segment_ids, dtype=torch.long),
                "valid_ids1": torch.tensor(valid_ids1, dtype=torch.long),
                "valid_ids2": torch.tensor(valid_ids2, dtype=torch.long),
                "input_text": example.input,
                "output_text": example.output
            })

        return features

    def get_dataloader(self, features, batch_size, mode='train', rank=0,  world_size=1):
        if mode == "train" and world_size > 1:
            features = features[rank::world_size]

        data_set = DiaDataset(features)
        sampler = RandomSampler(data_set)
        return DataLoader(data_set, sampler=sampler, batch_size=batch_size)

    def get_all_dataloader(self, tokenizer, args):
        #train
        train_examples = self.get_train_examples()
        train_features = self.convert_to_feature(tokenizer, train_examples, args.max_seq_len)
        train_dataloader = self.get_dataloader(train_features, mode="train", rank=args.rank,
                                                    world_size=args.world_size, batch_size=args.batch_size)

        #test
        test_examples = self.get_test_examples()
        test_features = self.convert_to_feature(tokenizer, test_examples, args.max_seq_len)
        test_dataloader = self.get_dataloader(test_features, mode="test", batch_size=args.batch_size)

        #dev
        dev_examples = self.get_dev_examples()
        dev_features = self.convert_to_feature(tokenizer, dev_examples, args.max_seq_len)
        dev_dataloader = self.get_dataloader(dev_features, mode="dev", batch_size=args.batch_size)

        return train_dataloader, dev_dataloader, test_dataloader