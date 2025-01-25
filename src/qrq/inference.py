import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.functional import softmax
from transformers import T5Tokenizer
import time
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


def run_instance_intent(batch, model, device='gpu'):
    logits = model(batch[0].long().to(device))
    labels = batch[1].to(device)
    logits = softmax(logits, dim=2)
    # print(logits[0][0], sum(logits[0][0]))
    logits = torch.argmax(logits, dim=2)
    # print(labels.shape, logits.shape)  
    precision_hist = []
    recall_hist = []    
    f1_hist = []    
    for j in range(labels.shape[0]):
        precision = precision_score(labels[j].cpu(), logits[j].cpu())
        recall = recall_score(labels[j].cpu(), logits[j].cpu())    
        f1 =  f1_score(labels[j].cpu(), logits[j].cpu())
        # print(precision, recall, f1)
        precision_hist.append(precision)
        recall_hist.append(recall)
        f1_hist.append(f1)
    print("Iteration: {}, precision: {:.4f}, recall: {:.4f}, F1: {:.4f}"
                    .format(j, np.mean(precision_hist), np.mean(recall_hist), np.mean(f1_hist)))        
    return [np.mean(precision), np.mean(recall_hist), np.mean(f1)]

def generate_instance(batch, model, tokenizer):
        out_tokens = model.generate(batch[0].long())[0]
        inp_tokens = batch[0][0]
        trg_tokens = batch[1][0]
        
        inp_text = tokenizer.convert_ids_to_tokens(inp_tokens[inp_tokens!=0])
        out_text = tokenizer.convert_ids_to_tokens(out_tokens[out_tokens!=0])
        trg_text = tokenizer.convert_ids_to_tokens(trg_tokens[trg_tokens!=0])
        print("input text: {}".format(' '.join(inp_text)))
        print("output text: {}".format(' '.join(out_text)))
        print("target text: {}".format(' '.join(trg_text)))
        print()

def calc_perplexity(model, data_loader):
    # See an explanation at https://huggingface.co/transformers/perplexity.html
    perplexity = []
    model.set_eval()

    for src, trg in data_loader:
        N, L = src.shape[0], src.shape[1]
        logits = model(src, trg)
        # logits: N x L x n_classes
        # Get the likelihood of the true targets
        likelihood = logits[trg]
        # likelihood: N x L
        log_likelihood = torch.log(likelihood)
        # log_likelihood: N x L
        PPL = torch.exp(-torch.mean(log_likelihood, dim=-1))
        # PPL: N
        batch_perplexity = torch.mean(PPL, dim=-1)
        # batch_perplexity: N
        perplexity.extend(batch_perplexity)

    return np.mean(np.array(perplexity))

def inference(data_loader, model, args):
    history = []
    start = time.time()

    if args.experiment == "intent_classifier":
        for i, batch in enumerate(data_loader):
            metrics = run_instance_intent(batch, model)
            history.append(metrics)
    if args.experiment == "enc_dec":
        tokenizer = T5Tokenizer.from_pretrained('t5-small')
        for i, batch in enumerate(data_loader):
            metrics = run_instance_enc_dec(batch, model, tokenizer, args)
            history.append(metrics)
    elif args.experiment == "qrq":
        pass
    return history

def generate(data_loader, model, args, num_samples=100):
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    for i, batch in enumerate(data_loader):
        generate_instance(batch, model, tokenizer)
        if i >= num_samples:
            break
