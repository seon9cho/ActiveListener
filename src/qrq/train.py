# Specific part of the project from Seong

import torch
import torch.nn as nn
from torch.autograd import Variable
import time
import numpy as np
import gc


def run_instance(batch, model, optimizer, device):
    loss = model(batch[0].to(device), batch[1].to(device))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss

def train(data_loader, model, optimizer, fname, args, valid_loader=None, print_every=100, save_every=5, validate_every=5, mode='intent'):
    train_history = []
    valid_history = []    
    start = time.time()
    model.to(args.device)
    for e in range(args.epochs):
        model.set_train(mode=mode)        
        temp_history = []
        for i, batch in enumerate(data_loader):
            loss = run_instance(batch, model, optimizer, args.device)
            temp_history.append(loss.item())
            if i % print_every == 0:
                checkpoint = time.time()
                train_history.append(np.mean(temp_history))
                print("Epoch: {}, Iteration: {}, loss: {:.4f}, elapsed: {:.2f}"
                      .format(e, i, np.mean(temp_history), checkpoint - start))
                temp_history = []
            del batch
            gc.collect()

        if e % save_every == 0:
            if args.save_model:
                torch.save(model.state_dict(), args.models_dir+fname +'.pt')
        if valid_loader is not None:
            if e % validate_every == 0:
                model.set_eval(mode=mode)
                for j, batch in enumerate(valid_loader):
                    metrics = run_instance_intent(batch, model, args.device)
                    valid_history.append(metrics)            
                    # print("Iteration: {}, precision: {:.4f}, recall: {:.4f}, F1: {:.4f}"
                    #     .format(j, metrics[0], metrics[1], metrics[2])) 
    return train_history, valid_history


# if __name__ == '__main__':
#     d_model = 256
#     intent_hidden_dim = 256
#     n_intent_classes = 10
#     vocab_size = len(word2index)
#     num_epochs = 10
#     batch_size = 16
#     model = QRQTransformer(vocab_size, d_model, intent_hidden_dim, n_intent_classes).to(device)
#     print("Model created.")
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)
#     loader = DataLoader(data, batch_size)
#     print("Start training.")
#     history = train(loader, model, criterion, optimizer, num_epochs)




