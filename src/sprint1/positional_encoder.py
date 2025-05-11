import torch.nn as nn
import torch
import math
import numpy as np
class PositionEncoding(nn.Module):
    def __init__(self,dim: int,max_len: int = 100):
        super().__init__()

        # This creates the tensor array of max_len*dim max_len will be rows and dim will be the columns.
        pe = torch.zeros(max_len,dim)

        print(pe)
        # The torch.arrange will give 1d tensor array from 0 to max_len-1 [1...max_len-1].
        # unsqueeze(0) will take list as 1 element -> [[1...max_len-1]].
        # unsqueeze(1) will convert each element and convert it to list -> [[1],...[max_len-1]].
        position = torch.arange(0,max_len).unsqueeze(1)
        print(position)
        div_term = torch.exp(torch.arange(0,dim,2)*-(math.log(10000.0)/dim))
        print(div_term)

        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)
        print(pe)

        self.pe = pe.unsqueeze(0)
        print(self.pe)
if __name__ == "__main__":
    a = PositionEncoding(8,16)
