# provides a sample input and passes it through the entire mock pipeline to produce a valid (although likely useless) output
import torch
import torch.nn as nn
from torch.autograd import Variable
import time
from transformers import T5Config, T5Model, T5EncoderModel, T5ForConditionalGeneration

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def run_instance(batch, criterion, model, optimizer):
    output = model(batch.src, batch.trg)
    loss = criterion(batch, output)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss

def train(data_loader, model, criterion, optimizer, num_epochs, print_every=100):
    history = []
    start = time.time()
#     try:
    for e in range(num_epochs):
        data_loader.shuffle()
        temp_history = []
        for i,batch in enumerate(data_loader.batches):
            loss = run_instance(batch, criterion, model, optimizer)
            temp_history.append(loss.item())
            if i % print_every == 0:
                checkpoint = time.time()
                history.append(np.mean(temp_history))
                print("Epoch: {}, Iteration: {}, loss: {:.4f}, elapsed: {:.2f}"
                      .format(e, i, np.mean(temp_history), checkpoint - start))
                temp_history = []
    return history

if __name__ == '__main__':
    d_model = 512
    intent_hidden_dim = 512
    n_intent_classes = 10
    vocab_size = 100
    num_epochs = 10
    batch_size = 16
    model = QRQTransformer(vocab_size, d_model, intent_hidden_dim, n_intent_classes)
    print("Model created.")
    sample_input = torch.randint(100, (10, 15))
    model.set_eval()
    logits = model.predict_intent(sample_input)
    print(logits.shape)
    model.set_train("intent")
    sample_label = torch.randint(2, (10, 10))
    print(sample_label)
    loss = model.predict_intent(sample_input, sample_label)
    print(loss)

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)
    # loader = DataLoader(data, batch_size)
    # print("Start training.")
    # history = train(loader, model, criterion, optimizer, num_epochs)
    # criterion1 = CrossEntropyLossIntentClass()
    # criterion2 = CrossEntropyLossEncDec()
    # criterion3 = CrossEntropyInformation()
    # sample_input1 = torch.tensor([0.9, 0.8, 0.7, 0.1, 0.2, 0.5])
    # sample_target1 = torch.tensor([1, 0, 0, 1, 0, 0])
    # loss1 = criterion1(sample_input, sample_target)
    # run_instance
    # print(loss1)




