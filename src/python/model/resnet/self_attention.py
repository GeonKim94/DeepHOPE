# original paper https://arxiv.org/abs/1805.08318 
# ref code https://velog.io/@choonsik_mom/SAGANSelf-Attention-GAN

import torch
import torch.nn as nn
from einops import rearrange

def SA_module(in_dim, at_dim, num_head = 4, compress_qk = 8):
    if num_head == 1:
        return Self_Attention(in_dim, at_dim, compress_qk)
    else:
        return Self_Attention_MH(in_dim, at_dim, num_head, compress_qk)


class Self_Attention_MH(nn.Module): # multihead
    def __init__(self,in_dim, at_dim, num_head = 4, compress_qk = 8):
        super(Self_Attention_MH, self).__init__()
        self.list_head = nn.ModuleList()
        for i in range(num_head):
            self.list_head.append(Self_Attention_MH_unit(in_dim, at_dim=at_dim // num_head, compress_qk=compress_qk))

    def forward(self, x):
        list_out = []
        for i in range(len(self.list_head)):
            out_ = self.list_head[i](x)
            list_out.append(out_)
        out = torch.cat(list_out, dim=1)
        out = out + x

        return out
    
    def _get_att(self, x):
        list_att = []
        list_val = []
        for i in range(len(self.list_head)):
            attention_map_T_, proj_value_ = self.list_head[i]._get_att(x)#out_, att_ = self.list_head[i](x)
            list_att.append(attention_map_T_.unsqueeze(-1))
            list_val.append(proj_value_.unsqueeze(-1))
        return torch.cat(list_att, dim=-1), torch.cat(list_val, dim=-1)

class Self_Attention_MH_unit(nn.Module) :
    def __init__(self, in_dim, at_dim, compress_qk) :
        super(Self_Attention_MH_unit, self).__init__()
        
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=at_dim//compress_qk,
                                   kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=at_dim//compress_qk,
                                 kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=at_dim//compress_qk,
                                   kernel_size=1)
    
        self.softmax = nn.Softmax(dim=-2)
        
        self.out_conv = nn.Conv2d(in_channels=at_dim//compress_qk, out_channels=at_dim,
                                   kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        X = x
        proj_query = rearrange(self.query_conv(X), 'b c h w -> b c (h w)') 
        proj_key = rearrange(self.key_conv(X), 'b c h w -> b c (h w)')
        proj_query = rearrange(proj_query, 'b c n -> b n c')
        S = torch.bmm(proj_query, proj_key)
        attention_map_T = self.softmax(S)
        proj_value = rearrange(self.value_conv(X), 'b c h w -> b c (h w)') 
        proj_qkv = torch.bmm(proj_value, attention_map_T)
        proj_qkv = proj_qkv.view(X.shape[0], self.out_conv.weight.shape[1], X.shape[2], X.shape[3])
        o = self.out_conv(proj_qkv)
        out = self.gamma * o
        return out

    def _get_att(self, x):
        X = x
        proj_query = rearrange(self.query_conv(X), 'b c h w -> b c (h w)')
        proj_key = rearrange(self.key_conv(X), 'b c h w -> b c (h w)')
        proj_query = rearrange(proj_query, 'b c n -> b n c')
        S = torch.bmm(proj_query, proj_key)
        attention_map_T = self.softmax(S)
        proj_value = rearrange(self.value_conv(X), 'b c h w -> b c (h w)')
        return attention_map_T, proj_value
    
class Self_Attention(nn.Module) :
    '''Self-Attention layer'''
    
    def __init__(self, in_dim, at_dim, compress_qk) :
        super(Self_Attention, self).__init__()
        
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=at_dim//compress_qk,
                                   kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=at_dim//compress_qk,
                                 kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=at_dim//compress_qk,
                                   kernel_size=1)
        
        self.softmax = nn.Softmax(dim=-2)
        
        
        self.out_conv = nn.Conv2d(in_channels=at_dim//compress_qk, out_channels=at_dim,
                                   kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x) :
        X = x
        proj_query = rearrange(self.query_conv(X), 'b c h w -> b c (h w)') 
        proj_key = rearrange(self.key_conv(X), 'b c h w -> b c (h w)')
        proj_query = rearrange(proj_query, 'b c n -> b n c')
        S = torch.bmm(proj_query, proj_key)
        attention_map_T = self.softmax(S) 
        proj_value = rearrange(self.value_conv(X), 'b c h w -> b c (h w)')
        proj_qkv = torch.bmm(proj_value, attention_map_T) 
        proj_qkv = proj_qkv.view(X.shape[0], self.out_conv.weight.shape[1], X.shape[2], X.shape[3])
        o = self.out_conv(proj_qkv)
        out = x + self.gamma * o

        return out

    
    
    def _get_att(self, x):
        X = x
        proj_query = rearrange(self.query_conv(X), 'b c h w -> b c (h w)') 
        proj_key = rearrange(self.key_conv(X), 'b c h w -> b c (h w)') 
        proj_query = rearrange(proj_query, 'b c n -> b n c') 
        S = torch.bmm(proj_query, proj_key)
        attention_map_T = self.softmax(S)
        proj_value = rearrange(self.value_conv(X), 'b c h w -> b c (h w)')
        return attention_map_T, proj_value
