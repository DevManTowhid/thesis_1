import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.model_utils import logits_to_prob

class PossibleSolutions():
    
    # FIX: Added context=None to __init__
    def __init__(self, tokens, log_probs, hidden, context=None):
        self.tokens = tokens
        self.log_probs = log_probs
        self.hidden = hidden
        self.context = context  # FIX: Store the context
        
    def extend(self, token, log_prob, hidden, context=None):
        return PossibleSolutions(
            tokens=self.tokens + [token], 
            log_probs=self.log_probs + [log_prob],
            hidden=hidden,
            context=context if context is not None else self.context # Keep old context if not updating
        )
    
    def n_gram_blocking(self, n):
        """ n-gramm blocking function preventing repeating identical sequence """
        return self.tokens[-n:]
    
    @property
    def latest_token(self):
        """ return last token for input of decoder """
        return self.tokens[-1]
    
    @property
    def avg_log_prob(self):
        """Estimating sequence average log probability to select best sequence at each beam step """
        return sum(self.log_probs) / len(self.tokens)
    

class BeamDecoder(nn.Module):
    
    def __init__(self, device, vocab, embeddings, decoder, beam_size=5, min_dec_steps=30, min_return_seq=1, num_return_sum=3, n_gram_block=3):
        super().__init__()
        
        self.device = device
        self.vocab = vocab
        self.beam_size = beam_size
        self.n_gram_block = n_gram_block
        self.min_dec_steps = min_dec_steps 
        self.min_return_seq = min_return_seq
        self.num_return_sum = num_return_sum
        self.output_dim = len(vocab)
        
        # Necessary models for decoding
        self.embedding = embeddings
        self.decoder = decoder
        
    def sort_hyps(self, hyps):
        """Sort hypotheses according to their log probability."""
        return sorted(hyps, key=lambda h: h.avg_log_prob, reverse=True)
    
    def filter_unk(self, idx):
        return idx if idx < len(self.vocab) else self.vocab.stoi["<unk>"]
    
    def decode(self, hidden, context, gen_len, batch_size):
        # hidden: [batch_size, hid_dim]
        # src_input: [seq_len, batch_size]
        best_hyps_all = []
        
        for idx in range(batch_size):
            hidden_idx = hidden[idx, :].unsqueeze(0) # [1, hid_dim]
            context_idx = context[idx, :].unsqueeze(0) # [1, hid_dim]
            
            # Creating hypotheses
            # FIX: Now this works because __init__ accepts context
            hyps = [PossibleSolutions(
                tokens=[self.vocab.start()],
                log_probs=[0.0],
                hidden=hidden_idx,
                context=context_idx,
            )]
            
            # Storing result for specific idx sentence
            sequence_results = []
            
            for t in range(gen_len):
                num_orig_hyps = len(hyps)
                dec_input = [self.filter_unk(hyp.latest_token) for hyp in hyps]
                dec_input = torch.tensor(dec_input, dtype=torch.long, device=self.device) # [K]
                dec_input = self.embedding(dec_input.unsqueeze(0)) # [1, K, emb_dim]
                
                hidden_idx = torch.cat([hyp.hidden for hyp in hyps], dim=0) # [K, hid_dim]
                # Gather contexts from hypotheses
                context_idx = torch.cat([hyp.context for hyp in hyps], dim=0) # [K, hid_dim]
                
                # Decoder block
                output_idx, hidden_idx = self.decoder(dec_input, hidden_idx.unsqueeze(0), context_idx.unsqueeze(0))
                log_prob = logits_to_prob(output_idx, method="softmax", tau=1.0, eps=1e-10, gumbel_hard=False)
            
                #Taking the best options through log distribution
                topk_probs, topk_ids = torch.topk(log_prob, k=self.beam_size*2, dim=-1) # [K, beam_size*2]
                
                # Beam decoding
                all_hyps = []
                for i in range(num_orig_hyps):
                    h_i = hyps[i]
                    hidden_i = hidden_idx[i].unsqueeze(0) # [1, hid_dim]
                    # FIX: Keep the context for the new hypothesis
                    context_i = context_idx[i].unsqueeze(0) 

                    for j in range(self.beam_size * 2):
                        if topk_ids[i, j].item() == self.vocab.unk():
                            pass
                        else:
                            if t > 0 and topk_ids[i, j].item() in h_i.n_gram_blocking(3):
                                    pass
                            else:
                                new_hyp = h_i.extend(
                                    token=topk_ids[i, j].item(), 
                                    log_prob=topk_probs[i, j].item(), 
                                    hidden=hidden_i,
                                    context=context_i # Pass context explicitly
                                )
                                all_hyps.append(new_hyp)
            
                hyps = [] 
                for hyp in self.sort_hyps(all_hyps):
                    if hyp.latest_token == self.vocab.stop():
                        if t >= self.min_dec_steps:
                            sequence_results.append(hyp)
                    else:
                        hyps.append(hyp)
                    if len(hyps) == self.beam_size or len(sequence_results) == self.beam_size:
                        break
            
                if len(sequence_results) == self.beam_size:
                    break
                
            if len(sequence_results) < self.min_return_seq:
                sequence_results = sequence_results + hyps[:self.min_return_seq - len(sequence_results)]
                
            sorted_results = self.sort_hyps(sequence_results)
            best_hyps = sorted_results[:self.min_return_seq]
            best_hyps_all.extend(best_hyps)
            
        hyp_words = [self.vocab.outputids2words(hyp.tokens) for i, hyp in enumerate(best_hyps_all)]
        hyp_results = [postprocess(words, skip_special_tokens=True, clean_up_tokenization_spaces=True) for words in hyp_words]
        
        return hyp_results

# Helper functions remain the same
def postprocess(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True):
    if skip_special_tokens:
        tokens = [t for t in tokens if not is_special(t)]
    out_string = ' '.join(tokens)
    if clean_up_tokenization_spaces:
        out_string = clean_up_tokenization(out_string)
    return out_string

def is_special(token):
    res = re.search("\<[a-z0-9]+\>", token)
    if res is None:
        return False
    return token == res.group()

def clean_up_tokenization(out_string):
    out_string = (
        out_string.replace(" .", ".")
        .replace(" ?", "?")
        .replace(" !", "!")
        .replace(" ,", ",")
        .replace(" ' ", "'")
        .replace(" n't", "n't")
        .replace(" 'm", "'m")
        .replace(" 's", "'s")
        .replace(" 've", "'ve")
        .replace(" 're", "'re")
    )
    return out_string
