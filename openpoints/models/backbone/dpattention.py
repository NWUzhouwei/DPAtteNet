import torch
import torch.nn as nn
from ..layers import create_act, create_grouper, furthest_point_sample, random_sample
import einops

class Attention(nn.Module):
    def __init__(self, in_planes, out_planes, group_args, group=8, dropout=0.3):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.group = group
        assert out_planes % group == 0
        
        self.linear_q = nn.Linear(in_planes, out_planes)
        self.linear_k = nn.Linear(in_planes, out_planes)
        self.linear_v = nn.Linear(in_planes, out_planes)
        
        self.grouper = create_grouper(group_args)
        
        self.center_pos_encode = nn.Sequential(           
            nn.Conv1d(3, out_planes, 1, bias=False), 
            nn.BatchNorm1d(out_planes),
            nn.ReLU(),
            nn.Conv1d(out_planes, out_planes, 1)
        )
        
        self.pos_encode = nn.Sequential(           
            nn.Conv2d(3, out_planes, 1, bias=False), 
            nn.BatchNorm2d(out_planes),
            nn.ReLU(),
            nn.Conv2d(out_planes, out_planes, 1)
        )
        
        self.weight_encode = nn.Sequential(           
            nn.Conv2d(out_planes, self.group, 1, bias=False), 
            nn.BatchNorm2d(self.group),
            nn.ReLU(),
            nn.Conv2d(self.group, self.group, 1)
        )
        
        self.softmax = nn.Softmax(dim=2)
        # self.drop = nn.Dropout(dropout)
        
    def forward(self, center_pos, center_fea,  pos, fea) -> torch.Tensor:
        '''
        center_pos: [b,m,3] / [b,n,3]
        center_fea: [b,c,m] / [b,c,n]
        pos: [b,n,3]
        fea: [b,c,n]
        '''
        x_q = self.linear_q(center_fea.transpose(-1, -2)) # [b,m,c]
        x_k = self.linear_k(fea.transpose(-1, -2)) # [b,n,c]
        x_v = self.linear_v(fea.transpose(-1, -2)) # [b,n,c]
        
        x_kv = torch.cat([x_k, x_v], dim=-1) # [b,n,2c]
        
        # ball query 相对坐标(/r) 特征
        neigh_pos, neigh_fea = self.grouper(center_pos, pos, x_kv.transpose(-1, -2).contiguous()) # (B, 3 + C, m, k)
        
        x_k, x_v = torch.chunk(neigh_fea.permute(0, 2, 3, 1), 2, dim=-1) # [b,m,k,c]
        
        center_p_r = self.center_pos_encode(center_pos.transpose(-1, -2)).transpose(-1, -2) # [b,m,c]
        p_r = self.pos_encode(neigh_pos).permute(0, 2, 3, 1) # [b,m,k,c]

        r_qk = x_k - x_q.unsqueeze(2) + center_p_r.unsqueeze(2) # [b,m,k,c]
        x_v = x_v + p_r # [b,m,k,c]
        
        w = self.weight_encode(r_qk.permute(0, 3, 1, 2)).permute(0, 2, 3, 1) # (n, nsample, c)
        w = self.softmax(w)
        # w = self.drop(w)
        
        # [b,m,k,g,i] i表示每组的特征通道数
        x_v = einops.rearrange(x_v, "b n k (g i) -> b n k g i", g=self.group)
        # [b,m,g,i]
        re = torch.einsum("b n k g i, b n k g -> b n g i", x_v, w)
        # [b,m,c]
        re = einops.rearrange(re, "b n g i -> b n (g i)")
        
        return re.transpose(-1, -2) # [b,c,m]

class TransformerBlock(nn.Module):
    def __init__(self, cin, cout, group_args, stride=1, sampler='fps', dropout=0.3):
        super().__init__()
        self.cin = cin
        self.cout = cout
        self.stride = stride
        if (stride > 1):
            if sampler.lower() == 'fps':
                self.sample_fn = furthest_point_sample
            elif sampler.lower() == 'random':
                self.sample_fn = random_sample
        self.atten = Attention(cin, cout, group_args)
        self.drop = nn.Dropout(dropout, inplace=True)
        if cin != cout:
            self.mapping = nn.Linear(cin, cout)

        self.mlp = MLP(in_features=cout, out_features=cout)
        
    def forward(self, xyz, fea):
        '''
        input: xyz: coords [b,n,3]
			   fea: feature [b,c,n]
	    out:   xyz: coords [b,n,3] / [b,m,3]
               fea: feature [b,c,n] / [b,c,m]
        '''
        _, c, n = fea.size()
        residual = fea
        layernorm1 = nn.LayerNorm([self.cin, n]).to('cuda')
        fea = layernorm1(fea)
        # 是否下采样
        if self.stride > 1: # 下采样
            m = xyz.shape[1] // self.stride
            idx = self.sample_fn(xyz, m).long() # [b,m]
            center_xyz = torch.gather(xyz, 1, idx.unsqueeze(-1).expand(-1, -1, 3)) # [b,m,3]
            center_fea = torch.gather(fea, -1, idx.unsqueeze(1).expand(-1, fea.shape[1], -1)) # [b,c,m]
            fea = self.atten(center_xyz, center_fea, xyz, fea) # [b,c,m]
            xyz = center_xyz # [b,m,3]
            residual = torch.gather(residual, -1, idx.unsqueeze(1).expand(-1, residual.shape[1], -1)) # [b,c,m]
            layernorm2 = nn.LayerNorm([self.cout, m]).to('cuda')
        else: # 不下采样
            fea = self.atten(xyz, fea, xyz, fea) # [b,c,n]
            layernorm2 = nn.LayerNorm([self.cout, n]).to('cuda')
        if self.cin != self.cout:
            residual = self.mapping(residual.transpose(-1, -2)).transpose(-1, -2)

        # res
        fea = residual + self.drop(fea)
        fea = fea + self.drop(self.mlp(layernorm2(fea).transpose(-1, -2))).transpose(-1, -2)
        return xyz, fea
    
class MLP(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x): # [b,n,c]
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x # [b,n,c]

class InvResAtten(nn.Module):
    def __init__(self,
                 in_channels,
                 norm_args=None,
                 act_args=None,
                 aggr_args={'feature_type': 'dp_fj', "reduction": 'max'},
                 group_args={'NAME': 'ballquery'},
                 conv_args=None,
                 expansion=1,
                 use_res=True,
                 num_posconvs=2,
                 less_act=False,
                 **kwargs
                 ):
        super().__init__()
        self.use_res = use_res
        self.tf = TransformerBlock(in_channels, in_channels, group_args)
        self.act = create_act(act_args)

    def forward(self, pf):
        p, f = pf
        identity = f
        _, f = self.tf(p, f.contiguous()) # [b,c,n]
        if f.shape[-1] == identity.shape[-1] and self.use_res:
            f += identity
        f = self.act(f)
        return [p, f]