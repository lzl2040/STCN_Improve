"""
network.py - The core of the neural network
Defines the structure and memory operations
Modifed from STM: https://github.com/seoungwugoh/STM

The trailing number of a variable usually denote the stride
e.g. f16 -> encoded features with stride 16
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.modules import *


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.compress = ResBlock(1024, 512)
        self.up_16_8 = UpsampleBlock(512, 512, 256) # 1/16 -> 1/8
        self.up_8_4 = UpsampleBlock(256, 256, 256) # 1/8 -> 1/4

        self.pred = nn.Conv2d(256, 1, kernel_size=(3,3), padding=(1,1), stride=1)

    def forward(self, f16, f8, f4):
        x = self.compress(f16)
        x = self.up_16_8(f8, x)
        x = self.up_8_4(f4, x)

        x = self.pred(F.relu(x))
        
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        return x


class MemoryReader(nn.Module):
    def __init__(self):
        super().__init__()
 
    def get_affinity(self, mk,qk,ms,qe):
        # B, CK, T, H, W = mk.shape
        # mk = mk.flatten(start_dim=2)
        # qk = qk.flatten(start_dim=2)
        #
        # # See supplementary material
        # a_sq = mk.pow(2).sum(1).unsqueeze(2)
        # ab = mk.transpose(1, 2) @ qk
        #
        # affinity = (2*ab-a_sq) / math.sqrt(CK)   # B, THW, HW
        #
        # # softmax operation; aligned the evaluation style
        # maxes = torch.max(affinity, dim=1, keepdim=True)[0]
        # x_exp = torch.exp(affinity - maxes)
        # x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
        # affinity = x_exp / x_exp_sum
        # 改成多头版本的
        B, CK, T, H, W = mk.shape
        ## 它不需要多头
        ms = ms.flatten(start_dim=1).unsqueeze(2) if ms is not None else None
        heads = 4
        ## 这些需要多头
        mk = mk.reshape(B,heads,-1,T,H,W)
        mk = mk.flatten(start_dim = 3)
        qk = qk.reshape(B,heads,-1,H,W)
        qk = qk.flatten(start_dim = 3)
        qe = qe.reshape(B,heads,-1,H,W)
        qe = qe.flatten(start_dim = 3) if qe is not None else None

        if qe is not None:
            # See appendix for derivation
            # or you can just trust me ヽ(ー_ー )ノ
            # mk = mk.transpose(1, 2)
            # a_sq = (mk.pow(2) @ qe)
            # two_ab = 2 * (mk @ (qk * qe))
            # b_sq = (qe * qk.pow(2)).sum(1, keepdim=True)
            mk = mk.transpose(2, 3)
            a_sq = (mk.pow(3) @ qe)
            two_ab = 2 * (mk @ (qk * qe))
            b_sq = (qe * qk.pow(3)).sum(2, keepdim=True)
            similarity = (-a_sq + two_ab - b_sq)
            # similarity = similarity.reshape(B,)
        else:
            # similar to STCN if we don't have the selection term
            a_sq = mk.pow(2).sum(1).unsqueeze(2)
            two_ab = 2 * (mk.transpose(1, 2) @ qk)
            similarity = (-a_sq + two_ab)
        ms = ms.unsqueeze(dim=1).expand(-1, heads, -1, -1)
        if ms is not None:
            similarity = similarity * ms / math.sqrt(CK)  # B*N*HW
        else:
            similarity = similarity / math.sqrt(CK)  # B*N*HW

        maxes = torch.max(similarity, dim=1, keepdim=True)[0]
        x_exp = torch.exp(similarity - maxes)
        x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
        affinity = x_exp / x_exp_sum
        return affinity

    def readout(self, affinity, mv, qv):
        heads = 4
        B, CV, T, H, W = mv.shape
        # mv = mv.reshape(B,heads,-1,T,H,W)

        mo = mv.view(B * heads, -1, T * H * W)
        affinity = affinity.view(B * heads, -1, H * W)
        #         print('mo shape:' + str(mo.shape))
        #         print('affinity shape:' + str(affinity.shape))
        mem = torch.bmm(mo, affinity)  # Weighted-sum B, CV, HW
        mem = mem.view(B, CV, H, W)

        mem_out = torch.cat([mem, qv], dim=1)

        return mem_out


class STCN(nn.Module):
    def __init__(self, single_object):
        super().__init__()
        self.single_object = single_object

        self.key_encoder = KeyEncoder()
        if single_object:
            self.value_encoder = ValueEncoderSO() 
        else:
            self.value_encoder = ValueEncoder() 

        # Projection from f16 feature space to key space
        self.key_proj = KeyProjection(1024, keydim=64)

        # Compress f16 a bit to use in decoding later on
        self.key_comp = nn.Conv2d(1024, 512, kernel_size=3, padding=1)

        self.memory = MemoryReader()
        self.decoder = Decoder()
        self.lfm = LFM(1024)

    def aggregate(self, prob):
        new_prob = torch.cat([
            torch.prod(1-prob, dim=1, keepdim=True),
            prob
        ], 1).clamp(1e-7, 1-1e-7)
        logits = torch.log((new_prob /(1-new_prob)))
        return logits

    def encode_key(self, frame): 
        # input: b*t*c*h*w
        b, t = frame.shape[:2]

        f16, f8, f4 = self.key_encoder(frame.flatten(start_dim=0, end_dim=1))
        f16 = self.lfm(f16)
        k16,shrinkage,selection = self.key_proj(f16)
        f16_thin = self.key_comp(f16)

        # B*C*T*H*W
        k16 = k16.view(b, t, *k16.shape[-3:]).transpose(1, 2).contiguous()
        shrinkage = shrinkage.view(b, t, *shrinkage.shape[-3:]).transpose(1, 2).contiguous()
        selection = selection.view(b, t, *selection.shape[-3:]).transpose(1, 2).contiguous()

        # B*T*C*H*W
        f16_thin = f16_thin.view(b, t, *f16_thin.shape[-3:])
        f16 = f16.view(b, t, *f16.shape[-3:])
        f8 = f8.view(b, t, *f8.shape[-3:])
        f4 = f4.view(b, t, *f4.shape[-3:])

        return k16, f16_thin, f16, f8, f4,shrinkage,selection

    def encode_value(self, frame, kf16, mask, other_mask=None): 
        # Extract memory key/value for a frame
        if self.single_object:
            f16 = self.value_encoder(frame, kf16, mask)
        else:
            f16 = self.value_encoder(frame, kf16, mask, other_mask)
        return f16.unsqueeze(2) # B*512*T*H*W

    def segment(self, qk16, qv16, qf8, qf4, mk16, mv16, query_selection,memory_shrinkage,selector=None):
        # q - query, m - memory
        # qv16 is f16_thin above
        affinity = self.memory.get_affinity(mk16, qk16,memory_shrinkage,query_selection)
        
        if self.single_object:
            logits = self.decoder(self.memory.readout(affinity, mv16, qv16), qf8, qf4)
            prob = torch.sigmoid(logits)
        else:
            logits = torch.cat([
                self.decoder(self.memory.readout(affinity, mv16[:,0], qv16), qf8, qf4),
                self.decoder(self.memory.readout(affinity, mv16[:,1], qv16), qf8, qf4),
            ], 1)

            prob = torch.sigmoid(logits)
            prob = prob * selector.unsqueeze(2).unsqueeze(2)

        logits = self.aggregate(prob)
        prob = F.softmax(logits, dim=1)[:, 1:]

        return logits, prob

    def forward(self, mode, *args, **kwargs):
        if mode == 'encode_key':
            return self.encode_key(*args, **kwargs)
        elif mode == 'encode_value':
            return self.encode_value(*args, **kwargs)
        elif mode == 'segment':
            return self.segment(*args, **kwargs)
        else:
            raise NotImplementedError


