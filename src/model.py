import torch.nn as nn
import torch
import math
from fmrec import FMRec
import torch.nn.functional as F
import copy
import numpy as np
import torch as th

import time


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class Att_FM_model(nn.Module):
    def __init__(self, fm, args):
        super(Att_FM_model, self).__init__()
        self.emb_dim = args.hidden_size
        self.item_num = args.item_num
        self.item_embeddings = nn.Embedding(self.item_num, self.emb_dim)
        self.embed_dropout = nn.Dropout(args.emb_dropout)
        self.position_embeddings = nn.Embedding(args.max_len, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.dropout)
        self.fm = fm
        self.loss_ce = nn.CrossEntropyLoss()
        self.loss_ce_rec = nn.CrossEntropyLoss(reduction='none')
        self.loss_mse = nn.MSELoss()
        self.mask_ratio = args.mask_ratio


    def fm_pre(self, item_rep, tag_emb, mask_seq):
        x_0, decode_out, t_rf_expand, t, z0  = self.fm(item_rep, tag_emb, mask_seq)
        return x_0, decode_out, t_rf_expand, t, z0 

    def reverse(self, item_rep, z0, mask_seq):
        reverse_pre = self.fm.reverse_p_sample_rf(item_rep, z0, mask_seq)
        return reverse_pre

    def loss_rec(self, scores, labels):
        return self.loss_ce(scores, labels.squeeze(-1))

    def loss_fm_ce(self, rep_fm, labels):
        scores = torch.matmul(rep_fm, self.item_embeddings.weight.t())
        """
        ### norm scores
        item_emb_norm = F.normalize(self.item_embeddings.weight, dim=-1)
        rep_fm_norm = F.normalize(rep_fm, dim=-1)
        temperature = 0.07
        scores = torch.matmul(rep_fm_norm, item_emb_norm.t())/temperature
        """
        return self.loss_ce(scores, labels.squeeze(-1))

    def fm_rep_pre(self, rep_fm):
        scores = torch.matmul(rep_fm, self.item_embeddings.weight.t())
        return scores
    
    def loss_rmse(self, rep_fm, labels):
        rep_gt = self.item_embeddings(labels).squeeze(1)
        return torch.sqrt(self.loss_mse(rep_gt, rep_fm))
    
    def routing_rep_pre(self, rep_fm):
        item_norm = (self.item_embeddings.weight**2).sum(-1).view(-1, 1)  ## N x 1
        rep_norm = (rep_fm**2).sum(-1).view(-1, 1)  ## B x 1
        sim = torch.matmul(rep_fm, self.item_embeddings.weight.t())  ## B x N
        dist = rep_norm + item_norm.transpose(0, 1) - 2.0 * sim
        dist = torch.clamp(dist, 0.0, np.inf)
        
        return -dist

    def regularization_rep(self, seq_rep, mask_seq):
        seqs_norm = seq_rep/seq_rep.norm(dim=-1)[:, :, None]
        seqs_norm = seqs_norm * mask_seq.unsqueeze(-1)
        cos_mat = torch.matmul(seqs_norm, seqs_norm.transpose(1, 2))
        cos_sim = torch.mean(torch.mean(torch.sum(torch.sigmoid(-cos_mat), dim=-1), dim=-1), dim=-1)  ## not real mean
        return cos_sim

    def regularization_seq_item_rep(self, seq_rep, item_rep, mask_seq):
        item_norm = item_rep/item_rep.norm(dim=-1)[:, :, None]
        item_norm = item_norm * mask_seq.unsqueeze(-1)

        seq_rep_norm = seq_rep/seq_rep.norm(dim=-1)[:, None]
        sim_mat = torch.sigmoid(-torch.matmul(item_norm, seq_rep_norm.unsqueeze(-1)).squeeze(-1))
        return torch.mean(torch.sum(sim_mat, dim=-1)/torch.sum(mask_seq, dim=-1))
    
    def loss_FM_mse(self, rep_fm, target_embeddings):

        loss = self.loss_mse(rep_fm, target_embeddings)
        return loss

    def position_sincos_embedding(self, position_ids, dim, max_period=10000):
        device = position_ids.device
        position_ids = position_ids.float()

        dim_t = th.arange(dim, dtype=th.float32, device=device)
        dim_t = max_period ** (2 * (dim_t // 2) / dim)

        position = position_ids[:, None]  # [N, 1]
        div_term = position / dim_t  # [N, dim]

        embedding = th.zeros((position.size(0), dim), device=device)

        embedding[:, 0::2] = th.sin(div_term[:, 0::2])
        embedding[:, 1::2] = th.cos(div_term[:, 1::2])

        return embedding

    def switch_Matrix(self, sequence, device):

        batch_size, seq_len = sequence.size()

        num_items = self.item_num
        sparse_matrix = torch.zeros(batch_size, num_items, device=device)

        for i in range(batch_size):
            row_data = sequence[i]  
            non_zero_indices = row_data[row_data != 0]
            sparse_matrix[i, non_zero_indices] = 1

        return sparse_matrix

    def balanced_mse_loss(self, target, output, mask_ratio=1.0):

        target_shape = target.shape

        num_ones = torch.sum(target == 1).item()

        num_zeros = torch.sum(target == 0).item()

        num_selected_zeros = int(min(num_zeros, num_ones * mask_ratio))

        zero_positions = (target == 0).nonzero(as_tuple=True)
        one_positions = (target == 1).nonzero(as_tuple=True)

        zero_rows, zero_cols = zero_positions
        one_rows, one_cols = one_positions

        selected_zero_indices = torch.randint(0, num_zeros, (num_selected_zeros,))
        selected_zero_positions = (zero_rows[selected_zero_indices], zero_cols[selected_zero_indices])

        mask = torch.zeros_like(target)
        mask[selected_zero_positions] = 1
        mask[one_positions] = 1

        masked_target = target * mask
        masked_output = output * mask

        mse_loss = self.loss_mse(masked_output, masked_target)
        return mse_loss


    def forward(self, sequence, tag, forward_mse_time = 0, train_flag=True): 
        seq_length = sequence.size(1)

        item_embeddings = self.item_embeddings(sequence)
        item_embeddings = self.embed_dropout(item_embeddings)  ## dropout first than layernorm

        # position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)       ####
        # pos_embedding = self.position_sincos_embedding(position_ids, self.emb_dim)
        # position_embeddings = pos_embedding.unsqueeze(0).expand_as(item_embeddings)

        # position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        # position_embeddings = position_ids.unsqueeze(0).expand_as(sequence)
        # position_embeddings = self.position_embeddings(position_ids)

        item_embeddings = item_embeddings    ####

        item_embeddings = self.LayerNorm(item_embeddings)
        
        mask_seq = (sequence>0).float()

        # seq_Matrix = self.switch_Matrix(sequence, device=sequence.device)

        
        if train_flag:
            tag_emb = self.item_embeddings(tag.squeeze(-1))  ## B x H

            rep_fm, decode_out, t_rf_expand, t_rf, z0 = self.fm_pre(item_embeddings, tag_emb, mask_seq)

            seq_Matrix = self.switch_Matrix(sequence, device=sequence.device)

            loss_mse = self.balanced_mse_loss(seq_Matrix, decode_out, self.mask_ratio)

            ############X0_pred
            scores = loss_mse

            # ######V_pred
            # scores = tag_emb


            loss_FM_mse = self.loss_FM_mse(rep_fm, tag_emb)
            
            item_rep_dis = loss_FM_mse
            
        else:
            # noise_x_t = th.randn_like(tag_emb)
            z0 = th.randn_like(item_embeddings[:,-1,:])
            rep_fm = self.reverse(item_embeddings, z0, mask_seq)
            t_rf_expand, t_rf, item_rep_dis= None, None, None

            scores = None

        # item_rep = self.model_main(item_embeddings, rep_fm, mask_seq)
        # seq_rep = item_rep[:, -1, :]
        # scores = torch.matmul(seq_rep, self.item_embeddings.weight.t())
        

        # return scores, rep_fm, weights, t, item_rep_dis, seq_rep_dis

        return scores, rep_fm, t_rf_expand, t_rf, item_rep_dis, z0
        

def create_model_FM(args):
    FM_pre = FMRec(args)
    return FM_pre
