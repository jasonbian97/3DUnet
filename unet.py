
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


class TransitionDown(nn.Module):
    def __init__(self, drop_rate=0.0):
        super(TransitionDown, self).__init__()
        self.max_pool3d = nn.MaxPool3d( (2,2,2) )
    
    def forward(self, x):
        return self.max_pool3d(x)

class TransitionUp(nn.Module):
    def __init__(self, n_in_channels, n_out_channels):
        super(TransitionUp, self).__init__()
        #output_size = stride * (input_size-1) -2*padding + kernel_size + output_padding.  
        #e.g.  2*(32-1) - 2*1 + 3 +1 = 62 - 2 + 3 + 1
        self.transconv1 = nn.ConvTranspose3d(n_in_channels, n_out_channels, kernel_size=(3,3,3), stride=2, padding=(1,1,1), output_padding=1)
        self.n_out_channels = n_out_channels
    def forward(self, x):
        return self.transconv1(x)

class BasicBlock(nn.Module):
    def __init__(self, n_in_channels, n_out_channels, drop_rate=0.0, kernel_size=(3,3,3), training=True ):

        super(BasicBlock, self).__init__()

        padding = tuple([ int((k-1)/2) for k in kernel_size])
        self.conv1 = nn.Conv3d(n_in_channels, n_out_channels, 
                               kernel_size=kernel_size, stride=1,
                               padding=padding )

        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm3d(n_out_channels)
        self.drop_rate = drop_rate
        self.n_out_channels = n_out_channels

        self.drop3 = nn.Dropout3d(p=self.drop_rate)
        self.drop = nn.Dropout(p=self.drop_rate)
        self.training=training
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.drop_rate>0:
            x = self.drop(x)

            
        return x
        

class unet3(nn.Module):
    def __init__(self, n_channels, n_classes, drop_rate=0.0, training=True):
        super(unet3, self).__init__()
        
        self.drop_rate = drop_rate        

        # 1st layer down
        self.db1a = BasicBlock(n_channels,               2*n_channels )
        self.db1b = BasicBlock(self.db1a.n_out_channels, 2* self.db1a.n_out_channels)
        self.td1  = TransitionDown(drop_rate=drop_rate)

        # 2nd layer down
        self.db2a = BasicBlock(self.db1b.n_out_channels, 2*self.db1b.n_out_channels)
        self.db2b = BasicBlock(self.db2a.n_out_channels, 2*self.db2a.n_out_channels)
        self.td2  = TransitionDown(drop_rate=drop_rate)

        # 3rd layer down
        self.db3a = BasicBlock(self.db2b.n_out_channels, 2*self.db2b.n_out_channels)
        self.db3b = BasicBlock(self.db3a.n_out_channels, 2*self.db3a.n_out_channels)
        self.td3  = TransitionDown(drop_rate=drop_rate)

        # Bottleneck
        self.db4a = BasicBlock(self.db3b.n_out_channels, self.db3b.n_out_channels, drop_rate=drop_rate, training=training)
        self.db4b = BasicBlock(self.db4a.n_out_channels, self.db4a.n_out_channels )
        self.tu4 = TransitionUp(self.db4b.n_out_channels, self.db4b.n_out_channels)

        # 3rd layer up
        self.ub3a = BasicBlock(self.tu4.n_out_channels + self.db3b.n_out_channels, self.db3b.n_out_channels)
        self.ub3b = BasicBlock(self.ub3a.n_out_channels, self.ub3a.n_out_channels )
        self.tu3  = TransitionUp(self.ub3b.n_out_channels, self.ub3b.n_out_channels )

        # 2rd layer up
        self.ub2a = BasicBlock(self.tu3.n_out_channels + self.db2b.n_out_channels, self.db2b.n_out_channels)
        self.ub2b = BasicBlock(self.ub2a.n_out_channels, self.ub2a.n_out_channels )
        self.tu2  = TransitionUp(self.ub2b.n_out_channels, self.ub2b.n_out_channels )

        # 1st layer up
        self.ub1a = BasicBlock(self.tu2.n_out_channels + self.db1b.n_out_channels, self.db1b.n_out_channels)
        self.ub1b = BasicBlock(self.ub1a.n_out_channels, self.ub1a.n_out_channels )

        self.last_convolution = nn.Conv3d(self.ub1b.n_out_channels, n_classes, kernel_size=1)
        
        self.softmax_layer = nn.Softmax(dim=1)
        self.training=training



    def forward(self, x):        
        #DB0
        x  = self.db1a( x )
        x1 = self.db1b( x )
        #print(torch.cuda.max_memory_allocated(device=1)/1000000.0)
        x  = self.td1( x1 )
        #print(torch.cuda.max_memory_allocated(device=1)/1000000.0)
        x  = self.db2a( x )
        x2 = self.db2b( x )
        #print(torch.cuda.max_memory_allocated(device=1)/1000000.0)
        x  = self.td2( x2 )
        #print(torch.cuda.max_memory_allocated(device=1)/1000000.0)
        x  = self.db3a( x )
        x3 = self.db3b( x )
        #print(torch.cuda.max_memory_allocated(device=1)/1000000.0)
        x  = self.td3( x3 )
        #print(torch.cuda.max_memory_allocated(device=1)/1000000.0)
        x  = self.db4a( x )
        x  = self.db4b( x )
        #print(torch.cuda.max_memory_allocated(device=1)/1000000.0)
        x  = self.tu4( x )
        #print(torch.cuda.max_memory_allocated(device=1)/1000000.0)
        x  = torch.cat([x,x3],1)
        #print(torch.cuda.max_memory_allocated(device=1)/1000000.0)
        x  = self.ub3a( x )
        x  = self.ub3b( x )
        #print(torch.cuda.max_memory_allocated(device=1)/1000000.0)
        x  = self.tu3( x )
        #print(torch.cuda.max_memory_allocated(device=1)/1000000.0)
        x  = torch.cat([x,x2],1)
        #print(torch.cuda.max_memory_allocated(device=1)/1000000.0)
        x  = self.ub2a( x )
        x  = self.ub2b( x )
        #print(torch.cuda.max_memory_allocated(device=1)/1000000.0)
        x  = self.tu2( x )
        #print(torch.cuda.max_memory_allocated(device=1)/1000000.0)
        x  = torch.cat([x,x1],1)
        #print(torch.cuda.max_memory_allocated(device=1)/1000000.0)
        x  = self.ub1a( x )
        x  = self.ub1b( x )
        #print(torch.cuda.max_memory_allocated(device=1)/1000000.0)
        x = self.last_convolution(x)
        #print(torch.cuda.max_memory_allocated(device=1)/1000000.0)
        x = self.softmax_layer(x)
  
        return x

    def save(self,file_name):
        torch.save(self.state_dict(), file_name)

    def load(self,file_name):
        pretrained_dict = torch.load(file_name)
        self.load_state_dict( pretrained_dict )

    def load_cpu(self, file_name):
        pretrained_dict = torch.load(file_name,map_location='cpu')
        self.load_state_dict( pretrained_dict )

    def load_all_but_final_layer(self, file_name):
        cur_model_dict = self.state_dict()
        pretrained_dict = torch.load(file_name)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'last_convolution' not in k}
        cur_model_dict.update(pretrained_dict) 
        self.load_state_dict( cur_model_dict )


class unet3_4L(nn.Module):
    def __init__(self, n_channels, n_classes, drop_rate=0.0, training=True):
        super(unet3_4L, self).__init__()

        self.drop_rate = drop_rate

        # 1st layer down
        self.db1a = BasicBlock(n_channels, 2 * n_channels)
        self.db1b = BasicBlock(self.db1a.n_out_channels, 2 * self.db1a.n_out_channels)
        self.td1 = TransitionDown(drop_rate=drop_rate)

        # 2nd layer down
        self.db2a = BasicBlock(self.db1b.n_out_channels, 2 * self.db1b.n_out_channels)
        self.db2b = BasicBlock(self.db2a.n_out_channels, 2 * self.db2a.n_out_channels)
        self.td2 = TransitionDown(drop_rate=drop_rate)

        # 3rd layer down
        self.db3a = BasicBlock(self.db2b.n_out_channels, 2 * self.db2b.n_out_channels)
        self.db3b = BasicBlock(self.db3a.n_out_channels, 2 * self.db3a.n_out_channels)
        self.td3 = TransitionDown(drop_rate=drop_rate)

        # 4th layer down
        self.db4a = BasicBlock(self.db3b.n_out_channels, 2 * self.db3b.n_out_channels)
        self.db4b = BasicBlock(self.db4a.n_out_channels, 2 * self.db4a.n_out_channels)
        self.td4 = TransitionDown(drop_rate=drop_rate)

        # Bottleneck
        self.db5a = BasicBlock(self.db4b.n_out_channels, self.db4b.n_out_channels, drop_rate=drop_rate,
                               training=training)
        self.db5b = BasicBlock(self.db5a.n_out_channels, self.db5a.n_out_channels)
        self.tu5 = TransitionUp(self.db5b.n_out_channels, self.db5b.n_out_channels)

        # 4th layer up
        self.ub4a = BasicBlock(self.tu5.n_out_channels + self.db4b.n_out_channels, self.db4b.n_out_channels)
        self.ub4b = BasicBlock(self.ub4a.n_out_channels, self.ub4a.n_out_channels)
        self.tu4 = TransitionUp(self.ub4b.n_out_channels, self.ub4b.n_out_channels)

        # 3rd layer up
        self.ub3a = BasicBlock(self.tu4.n_out_channels + self.db3b.n_out_channels, self.db3b.n_out_channels)
        self.ub3b = BasicBlock(self.ub3a.n_out_channels, self.ub3a.n_out_channels)
        self.tu3 = TransitionUp(self.ub3b.n_out_channels, self.ub3b.n_out_channels)

        # 2rd layer up
        self.ub2a = BasicBlock(self.tu3.n_out_channels + self.db2b.n_out_channels, self.db2b.n_out_channels)
        self.ub2b = BasicBlock(self.ub2a.n_out_channels, self.ub2a.n_out_channels)
        self.tu2 = TransitionUp(self.ub2b.n_out_channels, self.ub2b.n_out_channels)

        # 1st layer up
        self.ub1a = BasicBlock(self.tu2.n_out_channels + self.db1b.n_out_channels, self.db1b.n_out_channels)
        self.ub1b = BasicBlock(self.ub1a.n_out_channels, self.ub1a.n_out_channels)

        self.last_convolution = nn.Conv3d(self.ub1b.n_out_channels, n_classes, kernel_size=1)

        self.softmax_layer = nn.Softmax(dim=1)
        self.training = training

    def forward(self, x):
        # DB0
        x = self.db1a(x)
        x1 = self.db1b(x)
        x = self.td1(x1)

        x = self.db2a(x)
        x2 = self.db2b(x)
        x = self.td2(x2)

        x = self.db3a(x)
        x3 = self.db3b(x)
        x = self.td3(x3)

        x = self.db4a(x)
        x4 = self.db4b(x)
        x = self.td4(x4)

        x = self.db5a(x)
        x = self.db5b(x)

        x = self.tu5(x)
        x = torch.cat([x, x4], 1)
        x = self.ub4a(x)
        x = self.ub4b(x)

        x = self.tu4(x)
        x = torch.cat([x, x3], 1)
        x = self.ub3a(x)
        x = self.ub3b(x)

        x = self.tu3(x)
        x = torch.cat([x, x2], 1)
        x = self.ub2a(x)
        x = self.ub2b(x)

        x = self.tu2(x)
        x = torch.cat([x, x1], 1)
        x = self.ub1a(x)
        x = self.ub1b(x)

        x = self.last_convolution(x)
        x = self.softmax_layer(x)

        return x

    def save(self, file_name):
        torch.save(self.state_dict(), file_name)

    def load(self, file_name):
        pretrained_dict = torch.load(file_name)
        self.load_state_dict(pretrained_dict)

    def load_cpu(self, file_name):
        pretrained_dict = torch.load(file_name, map_location='cpu')
        self.load_state_dict(pretrained_dict)

    def load_all_but_final_layer(self, file_name):
        cur_model_dict = self.state_dict()
        pretrained_dict = torch.load(file_name)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'last_convolution' not in k}
        cur_model_dict.update(pretrained_dict)
        self.load_state_dict(cur_model_dict)