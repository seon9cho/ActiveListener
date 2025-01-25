

import torch
import torch.nn as nn
from transformers import T5Config, T5ForConditionalGeneration

from .loss import IntentLoss, InformationGainLoss


class AttentionLayer(nn.Module):
    """Attention layer implemented as in self-attention, but with a trainable prototype query
    vector instead of a query that is a transformation of the input. Justification: for the 
    purposes of autoencoding and predicting, the prototype vector for summarizing the sequence 
    does not depend on different tokens - it is always has the same job: summarize the sequence 
    for e.g. an autoencoding task (as opposed to the job of predicting the next character, in 
    which the prototype vector changes based on the character preceding the character to predict.)
    """
    def __init__(self, d_model):
        super().__init__()

        # TODO: add multiple heads
        # The query should be a trainable prototype vector of weights, such that multiplying Q by K^T is
        # just multiplying K by a linear layer from d_model to 1
        self.lin_q = nn.Linear(d_model, 1)
        self.lin_k = nn.Linear(d_model, d_model)
        self.lin_v = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x, return_attention_weights=False):
        """
        Args:
            x ((N x L x d_model) torch.Tensor): the input embeddings
            return_attention_weights (bool): whether to return the tensor weights
                with the network output
        """
        d_model = x.shape[2]
        scale_factor = torch.sqrt(torch.tensor(d_model, dtype=torch.float)).to(x)
        # print(x.shape)
        k = self.lin_k(x)
        # k: N x L x d_model
        # This is where we differ from self-attention: we use a learnable prototype Q vector, 
        # implemented as a linear layer, instead of transforming the input to get queries
        attn = self.lin_q(k)
        # attn: N x L x 1
        attn = torch.transpose(attn, 1, 2)
        # attn: N x 1 x L
        attn = attn / scale_factor
        attn = self.softmax(attn)
        # attn: N x 1 x L
        v = self.lin_v(x)
        # v: N x L x d_model
        out = torch.bmm(attn, v).squeeze(1)
        # out: N x d_model

        if return_attention_weights:
            return out, attn.squeeze(1)
        
        return out

class IntentClassifier(nn.Module):
    def __init__(self, d_model=512, intent_hidden_dim=512, n_intent_classes=200):
        super().__init__()
        self.n_intent_classes = n_intent_classes

        # Predict n_intent_classes*2, because we have a binary classifier for each class
        self.model = nn.Sequential(
            AttentionLayer(d_model),
            nn.Linear(d_model, intent_hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(intent_hidden_dim),
            nn.Linear(intent_hidden_dim, intent_hidden_dim),
            nn.ReLU(),
            nn.Linear(intent_hidden_dim, n_intent_classes*2)
        )

    def forward(self, x):
        """
        Args:
            x ((N x L x d_model) torch.Tensor): the embedded input sentence
        Return:
            x ((N x n_classes x 2) torch.Tensor): the predictions for each class for each item in the batch
        """
        N = len(x)
        out = self.model(x)
        return out.view(N, self.n_intent_classes, 2)

class QRQTransformer(nn.Module):
    def __init__(self, intent_hidden_dim=512, n_intent_classes=200, max_length=512):
        super().__init__()
        self.max_length = max_length

        cfg = T5Config()
        self.t5_1 = T5ForConditionalGeneration(cfg).from_pretrained('t5-small')
        self.intent = IntentClassifier(d_model=cfg.d_model, intent_hidden_dim=intent_hidden_dim, n_intent_classes=n_intent_classes)
        self.t5_2 = T5ForConditionalGeneration(cfg).from_pretrained('t5-small')

        self.intent_loss = IntentLoss()
        # self.enc_dec_loss = nothing, because this is built into the T5 model
        self.info_gain_loss = InformationGainLoss()

    def set_train(self, mode):
        """Set the mode (intent classification vs. QRQ prediction) by freezing
        different parts of the model and changing the forward function.
        Args:
            mode (str): 'intent' or 'enc_dec' or 'qrq'
        """
        self.train = True
        self.mode = mode

        # Freeze all modules
        for param in self.t5_1.parameters():
            param.requires_grad = False
        for param in self.t5_2.parameters():
            param.requires_grad = False
        for param in self.intent.parameters():
            param.requires_grad = False
        # Put all modules in eval mode
        self.t5_1.train(False)
        self.t5_2.train(False)
        self.intent.train(False)

        if mode == 'intent':
            # Unfreeze the intent classifier
            for param in self.intent.parameters():
                param.requires_grad = True
            self.intent.train(True)
        elif mode == 'enc_dec':
            # Unfreeze the T5 decoder
            for param in self.t5_1.decoder.parameters():
                param.requires_grad = True
            self.t5_1.decoder.train(True)
        elif mode == 'qrq':
            # Unfreeze the T5 decoder
            for param in self.t5_1.decoder.parameters():
                param.requires_grad = True
            self.t5_1.decoder.train(True)
        else:
            raise ValueError(f"Mode {mode} is not a valid mode.")
    
    def set_eval(self, mode):
        self.train = False
        self.mode = mode
        
        # Freeze all modules
        for param in self.t5_1.parameters():
            param.requires_grad = False
        for param in self.t5_2.parameters():
            param.requires_grad = False
        for param in self.intent.parameters():
            param.requires_grad = False
        
        # Put all modules in eval mode
        self.t5_1.train(False)
        self.t5_2.train(False)
        self.intent.train(False)


    def predict_intent(self, x, label=None):
        """Use this version of the forward function to classify intent
        Args:
            x ((N x L) torch.Tensor): the tokenized query sequence
            label ((N x n_intent_classes) torch.Tensor): the class labels for each item
                in the batch.
        Returns:
            if self.train:
                ((1) torch.Tensor): the loss
            else:
                ((N x n_intent_classes x 2) torch.Tensor): predictions for each class
        """
        n = x.shape[0]
        dummy_decoder_input = torch.zeros(n, dtype=torch.long).unsqueeze(-1).to(next(self.parameters()).device)
        enc = self.t5_1(input_ids=x, decoder_input_ids=dummy_decoder_input).encoder_last_hidden_state
        #enc = torch.matmul(enc.permute(0, 2, 1), enc)
        logits = self.intent(enc)
        if self.train:
            loss = self.intent_loss(logits, label)
            return loss
        else:
            return logits

    def predict_enc_dec(self, src=None, trg=None):
        """Use this version of the forward function to predict the next character
        in the response
        Args:
            x ((N x L1) torch.Tensor): the tokenized query sequences - a list of 
                ids for each sequence
            label ((N x L2) torch.Tensor): the tokenized response sequences - a list of 
                ids for each sequence
        Returns:
            if self.train:
                ((1) torch.Tensor): the loss
            else:
                ((N x 1) torch.Tensor): the prediction for the next
                    token in the response
        """
        if self.train:
            # the forward function automatically creates the correct decoder_input_ids
            loss = self.t5_1(input_ids=src, labels=trg).loss
            return loss
        else:
            logits = self.t5_1(input_ids=src, decoder_input_ids=trg).logits
            return logits
            # https://huggingface.co/transformers/main_classes/model.html#transformers.generation_utils.GenerationMixin.generate
            # x = the prompt to start generation
            # return self.t5_1.generate(input_ids=x, max_length=50, num_beams=4, do_sample=True)

    def predict_qrq(self, src, trg=None):
        """Use this version of the forward function to get the information gain
        loss.
        Args:
            src ((N x L1) torch.Tensor): the tokenized query sequences - a list of 
                ids for each sequence
            trg (None): dummy label for uniformity
        Returns:
            if self.train:
                ((1) torch.Tensor): the loss
        """
        N = len(src)
        # Get ENC_1 hidden states
        enc_1_hidden_states = self.t5_1.encoder(input_ids=src)[0]
        # Get intent classification 1
        intent_1 = self.intent(enc_1_hidden_states)
        # Get DEC_1 logits/embeddings
        pad_token = self.t5_1.config.pad_token_id
        # Create an input embedding buffer for DEC_1
        dec_1_inp_embeddings = self.t5_1.shared(torch.tensor([pad_token]*N).to(src.device)).unsqueeze(1).expand(-1, self.max_length+1, -1)
        start_token = self.t5_1.config.decoder_start_token_id
        dec_1_inp_embeddings[:, 0, :] = self.t5_1.shared(torch.tensor([start_token]*N).to(src.device))
        dec_1_logits = torch.zeros(N, self.max_length, self.t5_1.config.vocab_size, dtype=torch.float32).to(src.device)
        for i in range(self.max_length):
            dec_outputs = self.t5_1.decoder(encoder_hidden_states=enc_1_hidden_states, inputs_embeds=dec_1_inp_embeddings[:, :i+1, :].clone())[0]
            # -1 because we only want the logits for the next token
            dec_1_logits[:, i, :] = self.t5_1.lm_head(dec_outputs)[:, -1, :]
            dec_1_inp_embeddings[:, i+1, :] = torch.matmul(dec_1_logits[:, i, :].clone(), self.t5_1.shared.weight)
        # Pass embeddings into ENC_2 to get ENC_2 hidden states
        enc_2_inp_embeddings = dec_1_inp_embeddings[:, 1:, :].clone()
        enc_2_hidden_states = self.t5_2.encoder(inputs_embeds=enc_2_inp_embeddings)[0]
        # Get DEC_2 logits/embeddings
        dec_2_inp_embeddings = self.t5_2.shared(torch.tensor([pad_token]*N).to(src.device)).unsqueeze(1).expand(-1, self.max_length+1, -1)
        dec_2_inp_embeddings[:, 0, :] = self.t5_2.shared(torch.tensor([start_token]*N).to(src.device))
        dec_2_logits = torch.zeros(N, self.max_length, self.t5_2.config.vocab_size, dtype=torch.float32).to(src.device)
        for i in range(self.max_length):
            dec_outputs = self.t5_2.decoder(encoder_hidden_states=enc_2_hidden_states, inputs_embeds=dec_2_inp_embeddings[:, :i+1, :].clone())[0]
            # -1 because we only want the logits for the next token
            dec_2_logits[:, i, :] = self.t5_2.lm_head(dec_outputs)[:, -1, :]
            dec_2_inp_embeddings[:, i+1, :] = torch.matmul(dec_2_logits[:, i, :].clone(), self.t5_2.shared.weight)
        # Pass embeddings into ENC_1 to get ENC_3 hidden states (ENC_1, ENC_2, and ENC_3 share the same weights)
        enc_3_inp_embeddings = dec_2_inp_embeddings[:, 1:, :].clone()
        enc_3_hidden_states = self.t5_1.encoder(inputs_embeds=enc_3_inp_embeddings)[0]
        # Get intent classification 2
        intent_2 = self.intent(enc_3_hidden_states)
        # Calculate loss
        loss = self.info_gain_loss(intent_1, intent_2)
        if self.train:
            return loss
        else:
            return loss, dec_1_logits, dec_2_logits, intent_1, intent_2
    
    def generate(self, src):
        if self.mode == 'enc_dec':
            return self.t5_1.generate(input_ids=src, max_length=50, num_beams=4, do_sample=True)

    def forward(self, x, label=None):
        """
        Case 1: mode is 'intent'
        Args:
            x ((N x L) torch.Tensor): the input query sequence ids
            label ((N x n_intent_classes) torch.Tensor): the class labels for each item
                in the batch.
        Case 2: mode is 'enc_dec':
        Args:
            x ((N x L) torch.Tensor): the input query sequence ids
            label ((N x n_intent_classes) torch.Tensor): the class labels for each item
                in the batch.
        """
        if self.mode == 'intent':
            return self.predict_intent(x, label)
        elif self.mode == 'enc_dec':
            return self.predict_enc_dec(x, label)
        elif self.mode == 'qrq':
            return self.predict_qrq(x, label)
            # return torch.random.randn()
            # return self.predict_info_gain(x)
        else:
            raise ValueError
