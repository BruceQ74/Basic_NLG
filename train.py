# coding = utf-8

from pytorch_transformers import BertTokenizer,BertModel

from baseline import EncoderDecoder

import logging
import argparse
import math
import os
import sys
from time import strftime, localtime
import random
import numpy as np

import torch

from data_utils import DataProcessor

from optimization import BertAdam, warmup_linear
from schedulers import LinearWarmUpScheduler, PolyWarmUpScheduler

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

CONFIG_NAME = 'config.json'
WEIGHTS_NAME = 'pytorch_model.bin'


def get_rank():
    import torch.distributed as dist
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0


class Instructor:
    def __init__(self, args):
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_model)
        self.data_processor = DataProcessor(args.data_dir, args.dataset)

        if args.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(device=args.device.index)))


    def saving_model(self, saving_model_path, model, optimizer):
        if not os.path.exists(saving_model_path):
            os.mkdir(saving_model_path)
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(saving_model_path, WEIGHTS_NAME)
        output_config_file = os.path.join(saving_model_path, CONFIG_NAME)
        output_optimizer_file = os.path.join(saving_model_path, "optimizer.pt")
        torch.save(model_to_save.state_dict(), output_model_file)
        with open(output_config_file, "w", encoding='utf-8') as writer:
            writer.write(model_to_save.config.to_json_string())
        torch.save({'optimizer': optimizer.state_dict(),
                    'master params': optimizer},
                   output_optimizer_file)
        output_args_file = os.path.join(saving_model_path, 'training_args.bin')
        torch.save(self.args, output_args_file)

    def load_model(self, model, optimizer, saving_model_path):
        output_model_file = os.path.join(saving_model_path, WEIGHTS_NAME)
        output_optimizer_file = os.path.join(saving_model_path, "optimizer.pt")
        #model
        checkpoint_model = torch.load(output_model_file, map_location="cpu")
        model.load_state_dict(checkpoint_model)
        #optimizer
        checkpoint_optimizer = torch.load(output_optimizer_file, map_location="cpu")

        optimizer.load_state_dict(checkpoint_optimizer["optimizer"])
        return model, optimizer

    def save_args(self):
        output_args_file = os.path.join(self.args.outdir, 'training_args.bin')
        torch.save(self.args, output_args_file)

    def evaluation(self, logs):
        metrics = {}
        for metric in logs[0].keys():
            metrics[metric] = sum([log[metric] for log in logs])/len(logs)

        return metrics

    def _train_single_step(self, model, optimizer, scheduler, train_data_loader, global_step, args):
        n_correct, n_total, loss_total = 0, 0, 0
        tr_loss = 0
        average_loss = 0
        logits_evaluation = []
        label_evaluation = []
        logs = []

        # switch model to training mode
        model.train()
        for i_batch, sample_batched in enumerate(train_data_loader):
            input_ids = sample_batched["input_ids"].to(self.args.device)
            output_ids = sample_batched["output_ids"].to(self.args.device)

            loss, logits = model(input_ids, output_ids)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            
            average_loss += loss
            if (i_batch + 1) % args.gradient_accumulation_steps == 0:

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            if global_step % self.args.log_step == 0:
                logger.info('global_step: {}, loss: {:.4f}, lr: {:.6f}'.format(global_step, average_loss / i_batch, optimizer.param_groups[0]['lr']))
        return global_step

    def _train(self, model, optimizer, scheduler, train_data_loader, dev_dataloader, test_dataloader):
        path = None

        global_step = 0
        num_of_no_improvement = 0

        args = self.args
        for epoch in range(self.args.num_epoch):
            logger.info('>' * 100)
            logger.info('epoch: {}'.format(epoch))

            global_step = self._train_single_step(model, optimizer, scheduler, train_data_loader, global_step, args)

            # dev_result = self._evaluate(model, dev_dataloader)
            # logger.info("Epoch {}, Dev results: {}".format(epoch, dev_result))

            test_result = self._evaluate(model, test_dataloader, mode = "test")
            # logger.info("Epoch {}, Test results: {}".format(epoch, test_result))
            logger.info("Epoch {}, Test dataset done.".format(epoch))
        
        return path

    def _evaluate(self, model, data_loader, mode = "train"):
        n_correct, n_total, loss_total = 0, 0, 0
        tr_loss = 0
        average_loss = 0
        logits_evaluation = []
        label_evaluation = []
        logs = []
        test_results = []

        # switch model to eval mode
        model.eval()
        with torch.no_grad():
            for i_batch, sample_batched in enumerate(data_loader):
                input_ids = sample_batched["input_ids"].to(self.args.device)
                output_ids = sample_batched["output_ids"].to(self.args.device)

                logits = model.generate(input_ids, output_ids, self.args.max_generation_len)

                if mode == "test":
                    test_results.append(logits)
                    label_evaluation.append(output_ids.view(-1).tolist())
    
        if mode == "test":
            with open("test_result.txt", "w") as f:
                for i_, j_ in zip(test_results, label_evaluation):
                    _i = []
                    _j = []
                    for k, l in zip(i_, j_):
                        if k != 0:
                            _i.append(self.tokenizer.convert_ids_to_tokens(k))
                        if l != 0:
                            _j.append(self.tokenizer.convert_ids_to_tokens(l))
                    generation = ' '.join(_i)
                    original = ' '.join(_j)
                    f.write("Original: {}\nGeneration: {}\n----------------------------------------\n".format(original, generation))

        return 0

    def prepare_model_optimizer(self):
        n_vocab = self.tokenizer.vocab_size
        model = EncoderDecoder(d_model = self.args.d_model, n_vocab = n_vocab, num_layers=self.args.num_layers)
        print("Model: Transformer")
        print("build model...")


        if self.args.resume is True:
            print("resume from: {}".format(self.args.resume_model))
            output_model_file = os.path.join(self.args.resume_model, WEIGHTS_NAME)
            checkpoint_model = torch.load(output_model_file, map_location="cpu")
            model.load_state_dict(checkpoint_model)
        else:
            # model._reset_params(self.args.initializer)
            pass

        model = model.to(self.args.device)

        train_data_loader, test_data_loader, dev_dataloader = \
            self.data_processor.get_all_dataloader(self.tokenizer, self.args)

        num_train_optimization_steps = int(
            len(train_data_loader) / self.args.gradient_accumulation_steps) * self.args.num_epoch

        print(
        "trainset: {}, batch_size: {}, gradient_accumulation_steps: {}, num_epoch: {}, num_train_optimization_steps: {}".format(
            len(train_data_loader) * self.args.batch_size, self.args.batch_size, self.args.gradient_accumulation_steps,
            self.args.num_epoch, num_train_optimization_steps))

        # Prepare optimizer
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        print("Number of parameters:", sum(p[1].numel() for p in param_optimizer if p[1].requires_grad))

        # optimizer = BertAdam(optimizer_grouped_parameters,
        #                      lr=self.args.learning_rate,
        #                      warmup=self.args.warmup_proportion,
        #                      t_total=num_train_optimization_steps)

        optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=self.args.learning_rate)

        scheduler = None

        if self.args.local_rank != -1:
            try:
                from apex.parallel import DistributedDataParallel as DDP
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            model = DDP(model, delay_allreduce=True)
        elif self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        if self.args.resume is True:
            print("resume from: {}".format(self.args.resume_model))
            output_optimizer_file = os.path.join(self.args.resume_model, "optimizer.pt")
            checkpoint_optimizer = torch.load(output_optimizer_file, map_location="cpu")

            optimizer.load_state_dict(checkpoint_optimizer["optimizer"])

        return model, optimizer, scheduler, train_data_loader, test_data_loader, dev_dataloader

    def run(self):
        self.save_args()
        model, optimizer, scheduler, train_data_loader, dev_dataloader, test_data_loader = self.prepare_model_optimizer()
        self._train(model, optimizer, scheduler, train_data_loader, dev_dataloader, test_data_loader)


def get_args():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='WN18RR', type=str)
    parser.add_argument('--data_dir', default='WN18RR', type=str)
    parser.add_argument('--encoder', default='transformer', type=str)
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default='1e-5', type=float)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--bert_dropout', default=0.2, type=float)
    parser.add_argument('--num_epoch', default=30, type=int)
    parser.add_argument('--patient', default=5, type=int)
    parser.add_argument('--num_layers', default=2, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--log_step', default=100, type=int)
    parser.add_argument('--log', default='log', type=str)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--d_model', default=512, type=int)
    parser.add_argument('--bert_dim', default=1024, type=int)
    parser.add_argument('--max_seq_len', default=32, type=int)
    parser.add_argument('--max_generation_len', default=32, type=int)
    parser.add_argument('--device', default=None, type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--bert_model', default='bert-large-uncased', type=str)
    parser.add_argument('--outdir', default='./results', type=str)
    parser.add_argument('--warmup_proportion', default=0.06, type=float)
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int)
    parser.add_argument('--loss_scale', default=0, type=int)
    parser.add_argument('--save', action='store_true', help="Whether to save model")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument("--rank", type=int, default=0, help="local_rank for distributed training on gpus")
    parser.add_argument("--world_size", type=int, default=1, help="local_rank for distributed training on gpus")
    parser.add_argument('--resume', action='store_true', help="whether load previous checkpint and start training")
    parser.add_argument('--resume_model', default='', type=str)
    args = parser.parse_args()

    args.initializer = torch.nn.init.xavier_uniform_

    return args

def main():
    args = get_args()

    import datetime
    now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    if not os.path.exists(args.outdir):
        try:
            os.makedirs(args.outdir)
        except Exception as e:
            print(str(e))
    args.outdir = os.path.join(args.outdir, "{}_bts_{}_ws_{}_lr_{}_warmup_{}_seed_{}_bert_dropout_{}_{}".format(
        args.dataset,
        args.batch_size,
        args.world_size,
        args.learning_rate,
        args.warmup_proportion,
        args.seed,
        args.bert_dropout,
        now_time
    ))
    if args.save:
        args.outdir = "{}_save".format(args.outdir)
    if not os.path.exists(args.outdir):
        try:
            os.makedirs(args.outdir)
        except Exception as e:
            print(str(e))

    output_args_file = os.path.join(args.outdir, 'training_args.bin')
    torch.save(args, output_args_file)

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()

    if not os.path.exists(args.log):
        os.makedirs(args.log)

    log_file = '{}/{}-{}.log'.format(args.log, args.dataset, strftime("%y%m%d-%H%M", localtime()))
    logger.addHandler(logging.FileHandler(log_file))

    ins = Instructor(args)
    ins.run()


if __name__ == '__main__':
    main()
