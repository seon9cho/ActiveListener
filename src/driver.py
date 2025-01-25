import argparse
from qrq.experiments import (
    intent_classifier_experiment, enc_dec_experiment, qrq_experiment,
    intent_classifier_inference, enc_dec_inference, qrq_inference_perplexity,
    enc_dec_generate, qrq_enc_dec_experiment)
import torch


parser = argparse.ArgumentParser()
parser.add_argument(
    '--experiment', help='The type of experiment to run.', 
    default='intent_classifier', type=str, 
    choices=['intent_classifier', 'enc_dec', 'qrq', 'qrq_enc_dec'])
parser.add_argument(
    '--mode', default='train', type=str, 
    choices=['train', 'evaluate', 'generate'])
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--epochs', default=1, type=int)
parser.add_argument('--intent_hidden_dim', default=512, type=int)
parser.add_argument('--base_dir', default='../dataset', type=str)
parser.add_argument('--sample_dataset', default=False, type=bool)
parser.add_argument('--device', default='cuda', type=str, choices=['cpu', 'cuda'])
parser.add_argument('--save_model', default=True, type=bool)
parser.add_argument('--models_dir', default='/tmp/', type=str)
parser.add_argument('--max_length', help="The max sequence length", default=64, type=int)


if __name__ == '__main__':
    args = parser.parse_args()
    # Making sure that cuda is avaliable after cuda has been selected
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device=='cuda' else torch.device("cpu")    
    if args.mode == 'train':
        if args.experiment == 'intent_classifier':
            intent_classifier_experiment(args)
        elif args.experiment == 'enc_dec':
            enc_dec_experiment(args)        
        elif args.experiment == 'qrq':
            qrq_experiment(args)
        elif args.experiment == 'qrq_enc_dec':
            qrq_enc_dec_experiment(args)
    elif args.mode == 'evaluate':
        if args.experiment == 'intent_classifier':
            intent_classifier_inference(args)
        elif args.experiment == 'enc_dec':
            enc_dec_inference_perplexity(args)
        elif args.experiment == 'qrq':
            qrq_inference_perplexity(args)
    elif args.mode == 'generate':
        if args.experiment == 'enc_dec':
            enc_dec_generate(args)
