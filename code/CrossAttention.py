import torch
import torch.nn as nn
from torch.nn import Linear,MultiheadAttention

class CrossAttention(nn.Module):
    def __init__(self, input_size, num_heads):
        super(CrossAttention, self).__init__()
        #MultiHead
        self.MultiHead_1 = MultiheadAttention(embed_dim=input_size, num_heads=num_heads)


    def forward(self, input1, input2):
        input1 = input1.unsqueeze(0).transpose(0, 1)
        input2 = input2.unsqueeze(0).transpose(0, 1)
        output_1, attention_weights_1 = self.MultiHead_1(input1, input2, input2)
        attention_weights_1=attention_weights_1.squeeze(0)
        output_1 = output_1.transpose(0, 1).squeeze(0)

        return output_1, attention_weights_1
