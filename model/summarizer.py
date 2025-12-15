import torch
import torch.nn as nn

from model.attention import Attention, GAT
from model.encoder import Encoder

from utils.model_utils import cosine_similarity, load_gloveembeddings, project_vector, cosine_similarity, postprocess, is_special, clean_up_tokenization, mask_range, logits_to_prob

import random
import copy
from collections import OrderedDict

from model.revgrad import GradReversal
from model.beam_decoder import BeamDecoder


class Summarizer(nn.Module):
    def __init__(self, encoder_rec, encoder_cls, decoder, classifier, embedding_rec, embedding_cls, grl, vocab, hp, n_docs=8):
        super().__init__()
        self.vocab = vocab
        self.hp = hp
        self.input_dim = len(vocab)
        self.output_dim = len(vocab)
        self.device = self.hp.device
        
        # Encoder
        self.gat_rec = GAT(device=self.device,
                           input_dim=self.hp.hid_dim * 2,
                           output_dim=self.hp.hid_dim * 2,
                           hidden_dim=self.hp.gat_hidden,
                           dropout=self.hp.gat_drop,
                           alpha=self.hp.gat_alpha,
                           num_heads=self.hp.gat_heads)
        self.encoder = Encoder(input_dim=self.input_dim,
                                   hid_dim=self.hp.hid_dim,
                                   emb_dim=self.hp.enc_emb_dim,
                                   gat_attn=self.gat_rec,
                                   num_layers=self.hp.enc_layers,
                                   dropout=self.hp.enc_drop)
        
        self.encoder_rec = encoder_rec
        self.encoder_cls = encoder_cls
        self.decoder = decoder
        self.classifier = classifier
        self.grl = grl
        self.embedding_rec = embedding_rec
        self.embedding_cls = embedding_cls
        self.num_tgr_docs_domains = self.hp.num_tgr_docs_domains
        
        # used to reduce the dimension of the classifier input
        if self.num_tgr_docs_domains > 1:
            self.fc = nn.Linear(self.hp.dec_hidden * self.num_tgr_docs_domains, self.hp.dec_hidden)
        
        if self.hp.combine_encs == "ff":
            self.combine_encs_net = nn.Sequential(OrderedDict([
                ('ln1', nn.LayerNorm(n_docs * self.hp.hid_dim)),
                ('fc1', nn.Linear(n_docs * self.hp.hid_dim, self.hp.hid_dim)),
                ('relu1', nn.ReLU()),
                ('ln2', nn.LayerNorm(self.hp.hid_dim)),
                ('fc2', nn.Linear(self.hp.hid_dim, self.hp.hid_dim))
            ]))
            
        ########### Initializing beam decoder ###########
        self.beam_size = self.hp.beam_size
        self.min_dec_steps = self.hp.min_dec_steps
        self.num_return_seq = self.hp.num_return_seq
        self.num_return_sum = self.hp.num_return_sum
        self.n_gram_block = self.hp.n_gram_block
        self.beam_decoder = BeamDecoder(
            self.device, 
            self.vocab, 
            self.embedding_rec, 
            self.decoder, 
            self.hp.beam_size, 
            self.hp.min_dec_steps, 
            self.hp.num_return_seq, 
            self.hp.num_return_sum, 
            self.hp.n_gram_block
        )    

    # FIX: Added src_senti argument here
    def classify(self, src_senti, cls_hidden, rec_hidden, alpha):
        # cls_hiddden: [batch_size, hid_dim]
        # rec_hidden: [batch_size, hid_dim]
        if not self.hp.use_cls:
            return None, rec_hidden, rec_hidden
        
        batch_size = rec_hidden.shape[0]

        # Concat target domain docs representations
        if self.num_tgr_docs_domains == 1:
            hiddens = sample_target_docs(src_senti, cls_hidden, batch_size, pool="opposite", concat=self.hp.concat_docs) # [batch_size, hid_dim]
        elif self.num_tgr_docs_domains == 2:
            same_hiddens = sample_target_docs(src_senti, cls_hidden, batch_size, pool="same", concat=self.hp.concat_docs)
            opp_hiddens = sample_target_docs(src_senti, cls_hidden, batch_size, pool="opposite", concat=self.hp.concat_docs)
            hiddens = torch.concat([opp_hiddens, same_hiddens], dim=1) # [batch_size, hid_dim * 2]
            hiddens = self.fc(hiddens) # [batch_size, hid_dim]
        # No concatenation
        else:
            hiddens = cls_hidden
            
        cls_hidden = hiddens
        
        # Gradient Reversal Layer
        if self.hp.use_grl:
            cls_hidden = self.grl(cls_hidden, alpha=alpha)  # [batch_size, hid_dim]
                
        # Classification
        cls_preds = self.classifier(cls_hidden) # [batch_size]
        
        # Projection Mechanism
        if self.hp.use_proj:
            h_hat = project_vector(rec_hidden, cls_hidden, self.device)  # [batch_size, dec_dim]
            h_tilt = project_vector(rec_hidden, rec_hidden - h_hat, self.device)  # [batch_size, dec_dim]
            # h_hat: domain-shared text feature representations after GRL
            # h_tilt: domain-specific text feature representations
            if self.hp.dec_hidden_type == 1:
                return cls_preds, h_hat, h_hat
            if self.hp.dec_hidden_type == 2:
                return cls_preds, h_tilt, h_hat
        
        return cls_preds, rec_hidden, rec_hidden
    
    def decode_reviews(self, rec_hidden, trg, revs=False, tf_ratio=None):
        # rec_hidden: [batch_size, hid_dim]
        # trg: [seq_len,batch_size]
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.output_dim
        
        # Decoding
        context = rec_hidden # [batch_size, hid_dim]
        hidden = context
        dec_input = trg[0,:] # First input token is the <sos> token
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        for t in range(1, trg_len):
            #insert input token embedding, previous hidden state and the context state
            dec_emb_input = self.embedding_rec(dec_input.unsqueeze(0)) # [1, batch_size, emb_dim]
            #receive output tensor (predictions) and new hidden state
            output, hidden = self.decoder(dec_emb_input, hidden.unsqueeze(0), context.unsqueeze(0))
            prob = logits_to_prob(output, method="softmax", tau=1.0, eps=1e-10, gumbel_hard=False)
            #place predictions in a tensor holding predictions for each token
            outputs[t] = prob
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < tf_ratio if tf_ratio else False
            #get the highest predicted token from our predictions
            top1 = prob.argmax(1)
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            dec_input = trg[t] if teacher_force else top1
        
        return outputs
    
    def decode_summaries(self, revs_hidden, hiddens, context, trg, src_len, gumbel_hard=True):
        vocab_size = self.output_dim
        # Set summary length to the mean review length in the batch
        avg_len = int(torch.ceil(torch.mean(src_len.float())))
        sum_len = min(max(avg_len *2, 20), 75)
        
        # Compute mean representation
        if self.hp.combine_encs == "ff":
            mean_hidden = hiddens.contiguous().view(-1) # [batch_size * hid_dim]
            mean_context = context.contiguous().view(-1) # [batch_size * hid_dim]
            
            mean_hidden = self.combine_encs_net(mean_hidden.unsqueeze(0)) # [1, hid_dim]
            mean_context = self.combine_encs_net(mean_context.unsqueeze(0)) # [1, hid_dim]
        elif self.hp.combine_encs == "mean":
            mean_hidden = torch.mean(hiddens, dim=0).unsqueeze(0) # [1, hid_dim]
            mean_context = torch.mean(context, dim=0).unsqueeze(0) # [1, hid_dim]
        else:
            raise Error(f"Concatenation method '{self.hp.combine_encs}' is not supported")
        
        sum_hidden = mean_hidden
        sum_context = mean_context
        
        sum_outputs = torch.zeros(sum_len, 1, vocab_size).to(self.device)
        sum_dec_input = trg[0, 0].unsqueeze(0)
        for t in range(1, sum_len):
            sum_dec_emb_input = self.embedding_rec(sum_dec_input.unsqueeze(0)) # [1, batch_size, emb_dim]
            #if gumbel_hard and t != 0:
            #    print(dec_input.shape, self.lm.embedding.weight.shape)
            #    input_emb = torch.matmul(dec_input, self.lm.embedding.weight)
            #    print(input_emb.shape)
            
            sum_output, sum_hidden = self.decoder(sum_dec_emb_input, sum_hidden.unsqueeze(0), sum_context.unsqueeze(0))
            prob = logits_to_prob(sum_output, method="gumbel", tau=1.0, eps=1e-10, gumbel_hard=True)
            sum_outputs[t] = prob
            top1 = prob.argmax(1)
            sum_dec_input = top1
        
        sum_ids = sum_outputs.permute(1, 2, 0).argmax(1) # [1, sum_len]
        return sum_ids
    # PASTE THIS INSIDE THE Summarizer CLASS in summarizer.py
    
    def forward(self, src_input, trg, src_len, tf_ratio=0.5, gumbel_hard=True):
        # 1. Embed and Encode
        emb_input_rec = self.embedding_rec(src_input) # [seq_len, batch_size, emb_dim]
        rec_hidden = self.encoder_rec(emb_input_rec, self.device) # [batch_size, hid_dim]

        # 2. Handle Projection (Disentanglement)
        # If projection is ON, we want to extract the NEUTRAL (content) part
        if self.hp.use_proj and self.hp.use_cls:
            # We need the classifier encoder to find the "Sentiment Direction"
            emb_input_cls = self.embedding_cls(src_input)
            cls_hidden = self.encoder_cls(emb_input_cls, self.device)
            
            # h_hat is the Neutral Content (Original - Sentiment)
            h_hat = project_vector(rec_hidden, cls_hidden, self.device)
            context = h_hat 
        else:
            # If projection is OFF, just use the raw hidden state
            context = rec_hidden

        # 3. Decode
        # Pass the context (neutral vector) to the decoder
        outputs = self.decode_reviews(context, trg, tf_ratio=tf_ratio)

        # 4. Cosine Similarity (Placeholder)
        # Procedures.py expects a second return value. 
        # We return a dummy value here because the main training signal comes from 'outputs'.
        cos_sim = torch.tensor(0.0, device=self.device)

        return outputs, cos_sim

# FIX: Added this helper function which was missing
def sample_target_docs(domain, hidden, batch_size, pool="same", concat=False):
    out_hidden = []
    for idx in range(batch_size):
        if pool == "opposite":
            h = hidden[domain != domain[idx]] # [num target docs in batch, dec_dim]
        elif pool == "same":
            h = hidden[domain == domain[idx]] # [num source docs in batch, dec_dim]
            h = h[[i != idx for i, _ in enumerate(h)]]  # [num source docs in batch - 1, dec_dim]
        else:
            raise ValueError(f"domain: '{pool}' is not recognized. expected: all, same, or opposite.")
        
        if concat:
            h = torch.mean(h, dim=0).unsqueeze(0)
        else:
            if h.size(0) > 0:
                r = random.randint(0, h.size(0) - 1)
                h = h[r].unsqueeze(0)
            else:
                # Fallback if no docs match the condition (rare)
                h = hidden[idx].unsqueeze(0)
        
        out_hidden.append(h)
    
    out_hidden = torch.vstack(out_hidden)
    return out_hidden
