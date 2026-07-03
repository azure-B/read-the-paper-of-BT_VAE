import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class BT_VAE(nn.Module):
    def __init__(self, z_dim=32):
        super().__init__()
        self.z_dim = z_dim

        # BT pre_trained
        # we want to matrix output
        # shape ( batch, 32, 105, 145 )
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size = 3 , stride = 2 , padding = 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size = 3, stride = 2, padding = 1),
            nn.ReLU(),
        )

        self.BT = nn.Conv2d(32, 32, kernel_size = 3, stride = 2, padding = 1)
        self.common = nn.Conv2d(32, 32, kernel_size = 3, stride = 2, padding = 1)

        def encode(self,x) :
            x_left = torch.rot90(x, k = 1, dims=( -2, -1 ))
            common = self.encoder(x)
            BT = self.encoder(x_left)

            common = self.common(common)
            BT = self.BT(BT)

            return common, BT
        
        # Correlation Matrix
        def get_CM(self, BT, common):
            shape = BT.shape[1]

            BT = BT.permute( 0, 2, 3, 1 )
            BT = BT.reshape( -1, shape )

            N = BT.shape[0]
            
            common = common.permute( 0, 2, 3, 1 )
            common = common.reshape( -1, shape )

            BT_norm = ( BT - BT.mean(0) ) / ( BT.std(0) + 1e-8 )
            common_norm = ( common - common.mean(0) ) / ( common.std(0) + 1e-8 )

            CM = torch.matmul( BT_norm.T , common_norm ) / N
            return CM

        def get_BT_VAE_Loss(CM):
            shape = CM.shape[0]
            eye = torch.eye(shape)
            loss = 0
            w = 0.2

            for i in range(shape):
                for j in range(shape):
                    if i==j :
                        loss += ( eye[i][j] - CM[i][j] ).pow(2) 
                    else : 
                        loss += ( eye[i][j] - CM[i][j] ).pow(2) * w

            return loss