import torch.nn as nn
import torch
import torch.nn.functional as F

class Swish(nn.Module):
	def __init__(self,inplace = True):
		super().__init__()
		self.inplace = inplace
  
	def forward(self, x):
		if self.inplace:
			x.mul_(torch.sigmoid(x))
			return x
		else:
			return x * torch.sigmoid(x)

class MultiHeadAtten(nn.Module):
    def __init__(self, in_channel, out_channel, head, drop_rate=0.2):
        super().__init__()
        self.head_dim = out_channel // head # out_channel务必要能被head整除
        self.head = head
        self.wq = nn.Linear(in_channel, out_channel) # out_channel=head * self.head_dim
        self.wk = nn.Linear(in_channel, out_channel)
        self.wv = nn.Linear(in_channel, out_channel)

        self.o_dense = nn.Linear(out_channel, out_channel)
        self.o_drop = nn.Dropout(drop_rate)

    def forward(self,q,k,v): # [b,n,k,c]

        # linear projection
        key = self.wk(k)    # [b,n,k,head * self.head_dim]
        value = self.wv(v)  # [b,n,k,head * self.head_dim]
        query = self.wq(q)  # [b,n,k,head * self.head_dim]

        # split by head
        query = self.split_heads(query) # [b,head,n,k,self.head_dim]
        key = self.split_heads(key)
        value = self.split_heads(value)
        context = self.scaled_dot_product_attention(query, key, value) # [b,n,k,head*self.head_dim]
        o = self.o_dense(context) # [b,n,k,head*self.head_dim]
        o = self.o_drop(o)
        return o # [b,n,k,out_channel]

    def split_heads(self, x): # [b,n,k,c]
        x = torch.reshape(x,(x.shape[0], x.shape[1], x.shape[2], self.head, self.head_dim)) # [b,n,k,head,self.head_dim]
        return x.permute(0,3,1,2,4) # [b,head,n,k,self.head_dim]
    
    def scaled_dot_product_attention(self, q, k, v): # [b,head,n,k,self.head_dim]
        dk = torch.tensor(k.shape[-1]).type(torch.float) # 一个head的特征维度self.head_dim
        score = torch.matmul(q,k.permute(0,1,2,4,3)) / (torch.sqrt(dk) + 1e-8) # [b,head,n,k,k]
        
        attention = F.softmax(score,dim=-1) # [b,head,n,k,k]
        context = torch.matmul(attention,v) # [b,head,n,k,self.head_dim]
        context = context.permute(0,2,3,1,4)     # [b,n,k,head,self.head_dim]
        context = context.reshape((context.shape[0], context.shape[1], context.shape[2], -1))  
        return context  # [b,n,k,head*self.head_dim]
    
class Pool(nn.Module):
    def __init__(self, dim, pool='mix'):
        super().__init__()
        self.pool = pool
        assert pool in ['mix', 'max', 'mean', 'sum']
        if pool == 'mix':
            self.l = nn.Linear(dim*2, dim)
        
    def forward(self, fea):
        '''
        [B,M,K,C]->[B,M,C]
        [B,N,C]->[B,C]
        '''
        if self.pool == 'mix':
            mean = torch.mean(fea, dim=-2)
            max = torch.max(fea, dim=-2)[0]
            re = self.l(torch.cat([mean, max], dim=-1))
        elif self.pool == 'mean':
            re = torch.mean(fea, dim=-2)
        elif self.pool == 'max':
            re = torch.max(fea, dim=-2)[0]
        elif self.pool == 'sum':
            re = torch.sum(fea, dim=-2)
        return re
    
class ConvBNSwish1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=False),
            nn.BatchNorm1d(out_channels),
            Swish()
        )

    def forward(self, x):
        return self.net(x)