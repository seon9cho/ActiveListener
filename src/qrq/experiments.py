from .dataset import *
from .model import QRQTransformer
from .train import train
from .inference import inference, generate
from .loss import IntentLoss
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Setting seed for torch and numpy
torch.manual_seed(0)
np.random.seed(0)


def enc_dec_experiment(args):
    """Run the encoder-decoder experiment with the arguments specified in args
    Args:
        args (Namespace): the arguments for the experiment
    """
    n_intent_classes = 90   # This is really weired. How come the itent classes are reduced to 90 from 122
    print("Creating model...")
    model = QRQTransformer(intent_hidden_dim=args.intent_hidden_dim, n_intent_classes=n_intent_classes).to(args.device)    
    print("Model created.")
    print("Creating dataset...")
    # New dataset for QRQ
    train_dataset = QR_dataset(args.base_dir)

    train_loader = DataLoader(
         train_dataset,
         batch_size=args.batch_size,
         pin_memory=True,
         collate_fn=padding_collate_fn_enc_dec,
         shuffle=True
    )
    print("Dataset created.")
    # Training
    model.set_train(mode="enc_dec")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)    
    print("Begin training...")
    model_fname = 'enc_dec'

    # This training loop doesn't work for qrq experiment with RunTimeErro on line number 221 in model.py
    train(
        train_loader, model, optimizer, model_fname, args, None, mode='enc_dec')

def enc_dec_generate(args):
    n_intent_classes = 90
    model_fname = "enc_dec"
    model = QRQTransformer(intent_hidden_dim=args.intent_hidden_dim, n_intent_classes=n_intent_classes).to(args.device)
    print(args.models_dir + model_fname +'.pt')
    model.load_state_dict(torch.load(args.models_dir + model_fname +'.pt'))
    model.set_eval(mode="enc_dec")
    test_dataset = QR_dataset(args.base_dir)

    test_loader = DataLoader(
         test_dataset,
         batch_size=args.batch_size,
         pin_memory=True,
         collate_fn=padding_collate_fn_enc_dec,
         shuffle=True
    )
    generate(test_loader, model, args)

def qrq_enc_dec_experiment(args):
    """Run the encoder-decoder experiment with the arguments specified in args
    Args:
        args (Namespace): the arguments for the experiment
    """
    n_intent_classes = 90   # This is really weired. How come the itent classes are reduced to 90 from 122
    print("Creating model...")
    model = QRQTransformer(intent_hidden_dim=args.intent_hidden_dim, n_intent_classes=n_intent_classes).to(args.device)    
    print("Model created.")
    print("Creating dataset...")
    # New dataset for QRQ
    train_dataset = QRQ2Intent(args.base_dir, sample=args.sample_dataset)
    print(train_dataset[0])
    return

    train_loader = DataLoader(
         train_dataset,
         batch_size=args.batch_size,
         pin_memory=True,
         collate_fn=padding_collate_fn_enc_dec,
         shuffle=True
    )
    print("Dataset created.")
    # Training
    model.set_train(mode="enc_dec")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)    
    print("Begin training...")
    model_fname = 'enc_dec'

    # This training loop doesn't work for qrq experiment with RunTimeErro on line number 221 in model.py
    train(
        train_loader, model, optimizer, model_fname, args, None, mode='enc_dec')

def qrq_experiment(args):
    """Run the question-response-question experiment with the arguments specified in args
    Args:
        args (Namespace): the arguments for the experiment
    """
    n_intent_classes = 90   # This is really weired. How come the itent classes are reduced to 90 from 122
    print("Creating model...")
    model = QRQTransformer(intent_hidden_dim=args.intent_hidden_dim, n_intent_classes=n_intent_classes, max_length=args.max_length).to(args.device)    
    print("Model created.")
    print("Creating dataset...")
    # New dataset for QRQ
    train_dataset = QRQ2Intent(args.base_dir, sample=args.sample_dataset)

    train_loader = DataLoader(
         train_dataset,
         batch_size=args.batch_size,
         pin_memory=True,
         collate_fn=padding_collate_fn
    )
    print("Dataset created.")
    # Training
    model.set_train(mode="qrq")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)    
    print("Begin training...")
    model_fname = 'qrq'

    # This training loop doesn't work for qrq experiment with RunTimeErro on line number 221 in model.py
    train(
        train_loader, model, optimizer, model_fname, args, None, mode='qrq')
    if args.save_model:
        torch.save(model.state_dict(), args.models_dir+model_fname +'.pt')


def intent_classifier_experiment(args):
    """Run the intent classifier experiment with the arguments specified in args
    Args:
        args (Namespace): the arguments for the experiment
    """
    n_intent_classes = 90
    print("Creating model...")
    model = QRQTransformer(intent_hidden_dim=args.intent_hidden_dim, n_intent_classes=n_intent_classes).to(args.device)    
    print("Model created.")
    print("Creating dataset...")
    # base_dir = '../dataset'
    dataset = IntentDataset(args.base_dir, sample=args.sample_dataset)
    # batch_size = 2
    n_seq = len(dataset)
    # # Make train/val/test split
    # indices = np.arange(n_seq)
    train_val_split_loc = int(0.8 * n_seq)
    val_test_split_loc = n_seq - train_val_split_loc
    # train_indices, val_indices, test_indices = np.split(indices[torch.randperm(n_seq)], [train_val_split_loc, val_test_split_loc])
    # train_indices, val_indices, test_indices = set(train_indices), set(val_indices), set(test_indices)
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_val_split_loc, val_test_split_loc])
    train_loader = DataLoader(
         train_dataset,
         batch_size=args.batch_size,
         pin_memory=True,
         collate_fn=padding_collate_fn
    )

    valid_loader = DataLoader(
         valid_dataset,
         batch_size=args.batch_size,
         pin_memory=True,
         collate_fn=padding_collate_fn
    )    
    print("Dataset created.")
    # Training
    model.set_train(mode="intent")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)    
    print("Begin training...")
    model_fname = 'intent'
    train(
        train_loader, model, optimizer, model_fname, args, valid_loader, mode='intent')


def intent_classifier_inference(args):
    n_intent_classes = 90
    model_fname = 'intent'
    model = QRQTransformer(intent_hidden_dim=args.intent_hidden_dim, n_intent_classes=n_intent_classes).to(args.device)    
    print(args.models_dir + model_fname +'.pt')
    model.load_state_dict(torch.load(args.models_dir + model_fname +'.pt'))
    model.set_eval(mode="intent")
    test_dataset = IntentDataset(args.base_dir, sample=args.sample_dataset)

    test_loader = DataLoader(
         test_dataset,
         batch_size=args.batch_size,
         pin_memory=True,
         collate_fn=padding_collate_fn
    )
    model.to(args.device)
    inference(test_loader, model, args)

def enc_dec_inference(args):
    n_intent_classes = 90
    model_fname = "enc_dec"
    model = QRQTransformer(intent_hidden_dim=args.intent_hidden_dim, n_intent_classes=n_intent_classes).to(args.device)
    print(args.models_dir + model_fname +'.pt')
    model.load_state_dict(torch.load(args.models_dir + model_fname +'.pt'))
    model.set_eval(mode="enc_dec")
    test_dataset = QR_dataset(args.base_dir)

    test_loader = DataLoader(
         test_dataset,
         batch_size=args.batch_size,
         pin_memory=True,
         collate_fn=padding_collate_fn_enc_dec,
         shuffle=True
    )
    inference(test_loader, model, args)

def enc_dec_inference_perplexity(args):
    n_intent_classes = 90
    model_fname = "enc_dec"
    model = QRQTransformer(intent_hidden_dim=args.intent_hidden_dim, n_intent_classes=n_intent_classes).to(args.device)
    print(args.models_dir + model_fname +'.pt')
    model.load_state_dict(torch.load(args.models_dir + model_fname +'.pt'))
    model.set_eval(mode="enc_dec")
    test_dataset = QR_dataset(args.base_dir)

    test_loader = DataLoader(
         test_dataset,
         batch_size=args.batch_size,
         pin_memory=True,
         collate_fn=padding_collate_fn_enc_dec,
         shuffle=True
    )
    ppl = calc_perplexity(model, test_loader)
    print("PPL:", ppl)

def qrq_inference_perplexity(args):
    n_intent_classes = 90
    model_fname = "qrq"
    model = QRQTransformer(intent_hidden_dim=args.intent_hidden_dim, n_intent_classes=n_intent_classes).to(args.device)
    print(args.models_dir + model_fname +'.pt')
    model.load_state_dict(torch.load(args.models_dir + model_fname +'.pt'))
    model.set_eval(mode="qrq")
    test_dataset = QR_dataset(args.base_dir)

    test_loader = DataLoader(
         test_dataset,
         batch_size=args.batch_size,
         pin_memory=True,
         collate_fn=padding_collate_fn_enc_dec,
         shuffle=True
    )
    ppl = calc_perplexity(model, test_loader)
    print("PPL:", ppl)
