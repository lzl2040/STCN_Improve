import math
import torch


def softmax_w_top(x, top):
    values, indices = torch.topk(x, k=top, dim=1)
    x_exp = values.exp_()

    x_exp /= torch.sum(x_exp, dim=1, keepdim=True)
    # The types should be the same already
    # some people report an error here so an additional guard is added
    x.zero_().scatter_(1, indices, x_exp.type(x.dtype)) # B * THW * HW

    return x


class MemoryBank:
    def __init__(self, k, top_k=20):
        self.top_k = top_k

        self.CK = None
        self.CV = None

        self.mem_k = None
        self.mem_v = None
        self.mem_s = None

        self.num_objects = k
    
    def _do_softmax(self,similarity, top_k, inplace=False):
        # normalize similarity with top-k softmax
        # similarity: B x N x [HW/P]
        # use inplace with care
        if top_k is not None:
            values, indices = torch.topk(similarity, k=top_k, dim=1)

            x_exp = values.exp_()
            x_exp /= torch.sum(x_exp, dim=1, keepdim=True)
            if inplace:
                similarity.zero_().scatter_(1, indices, x_exp) # B*N*HW
                affinity = similarity
            else:
                affinity = torch.zeros_like(similarity).scatter_(1, indices, x_exp) # B*N*HW
        else:
            maxes = torch.max(similarity, dim=1, keepdim=True)[0]
            x_exp = torch.exp(similarity - maxes)
            x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
            affinity = x_exp / x_exp_sum 
            indices = None
        return affinity

    def _global_matching(self, mk, qk,ms,qe):
        # NE means number of elements -- typically T*H*W
        CK = mk.shape[1]
#         mk = mk.flatten(start_dim=2)
#         ms = ms.flatten(start_dim=1).unsqueeze(2) if ms is not None else None
#         qk = qk.flatten(start_dim=2)
#         qe = qe.flatten(start_dim=2) if qe is not None else None
        ms = ms.transpose(1,2)

        if qe is not None:
            # See appendix for derivation
            # or you can just trust me ヽ(ー_ー )ノ
            mk = mk.transpose(1, 2)
            a_sq = (mk.pow(2) @ qe)
            two_ab = 2 * (mk @ (qk * qe))
            b_sq = (qe * qk.pow(2)).sum(1, keepdim=True)
            similarity = (-a_sq+two_ab-b_sq)
        else:
            # similar to STCN if we don't have the selection term
            a_sq = mk.pow(2).sum(1).unsqueeze(2)
            two_ab = 2 * (mk.transpose(1, 2) @ qk)
            similarity = (-a_sq+two_ab)
        
        # print('similarity shape:' + str(similarity.shape))

        if ms is not None:
            similarity = similarity * ms / math.sqrt(CK)   # B*N*HW
        else:
            similarity = similarity / math.sqrt(CK)   # B*N*HW

        return similarity

    def _readout(self, affinity, mv):
        # print('affinity shape:' + str(affinity.shape))
        affinity = self._do_softmax(affinity,self.top_k)
        return torch.bmm(mv, affinity)

    def match_memory(self, qk,qe):
        k = self.num_objects
        _, _, h, w = qk.shape

        qk = qk.flatten(start_dim=2)
        qe = qe.flatten(start_dim=2)
        
        if self.temp_k is not None:
            mk = torch.cat([self.mem_k, self.temp_k], 2)
            mv = torch.cat([self.mem_v, self.temp_v], 2)
            ms = torch.cat([self.mem_s, self.temp_s], 2)
        else:
            mk = self.mem_k
            mv = self.mem_v
            ms = self.mem_s
        
#         print('mk shape:' + str(mk.shape))
#         print('qk shape:' + str(qk.shape))
#         print('ms shape:' + str(ms.shape))
#         print('qe shape:' + str(qe.shape))

        affinity = self._global_matching(mk,qk,ms,qe)

        # One affinity for all
        readout_mem = self._readout(affinity.expand(k,-1,-1), mv)

        return readout_mem.view(k, self.CV, h, w)

    def add_memory(self, key, value, memory_s,is_temp=False):
        # Temp is for "last frame"
        # Not always used
        # But can always be flushed
        self.temp_k = None
        self.temp_v = None
        self.temp_s = None
        key = key.flatten(start_dim=2)
        value = value.flatten(start_dim=2)
        memory_s = memory_s.flatten(start_dim=2)

        if self.mem_k is None:
            # First frame, just shove it in
            self.mem_k = key
            self.mem_v = value
            self.mem_s = memory_s
            self.CK = key.shape[1]
            self.CV = value.shape[1]
        else:
            if is_temp:
                self.temp_k = key
                self.temp_v = value
                self.temp_s = memory_s
            else:
                self.mem_k = torch.cat([self.mem_k, key], 2)
                self.mem_v = torch.cat([self.mem_v, value], 2)
                self.mem_s = torch.cat([self.mem_s,memory_s],2)