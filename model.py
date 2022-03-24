from torch import nn
import torch

class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Block, self).__init__()
        inter_ch = out_ch if in_ch > out_ch else out_ch // 2
        self.blk = nn.Sequential(
            nn.Conv3d(in_ch, inter_ch, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(inter_ch, out_ch, 3, stride=1, padding=1),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.blk(x)

class Encoder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Encoder, self).__init__()
        self.enc_blocks = Block(in_ch, out_ch)
        self.pool = nn.MaxPool3d(2)
    
    def forward(self, x):
        x = self.enc_blocks(x)
        pool = self.pool(x)
        return x, pool

class Decoder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Decoder, self).__init__()
        self.in_ch = in_ch
        self.dec_blocks = Block(in_ch//2+in_ch, out_ch)
        self.upsample = nn.ConvTranspose3d(in_ch, in_ch, 2, stride=2)

        
    def forward(self, x, encoder_feature):
        x = self.upsample(x)
        _, _, H, W, D = x.shape
        _, _, h, w, d = encoder_feature.shape
        height = (h-H)//2
        width = (w-W)//2
        depth = (d-D)//2
        encoder_feature = encoder_feature[:, :, height:height+H, width:width+W, depth:depth+D]
        x = torch.cat([x,encoder_feature], dim=1)
        x = self.dec_blocks(x)
        return x



class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.ch = [64, 128, 256, 512]
        self.enc1 = Encoder(1, self.ch[0])
        self.enc2 = Encoder(self.ch[0], self.ch[1])
        self.enc3 = Encoder(self.ch[1], self.ch[2])
        self.enc4 = Encoder(self.ch[2], self.ch[3])

        self.dec1 = Decoder(self.ch[3], self.ch[2])
        self.dec2 = Decoder(self.ch[2], self.ch[1])
        self.dec3 = Decoder(self.ch[1], self.ch[0])
        self.out_conv = nn.Conv3d(self.ch[0], 1, 1)

    def forward(self, x):
        enc_f1, x = self.enc1(x)
        enc_f2, x = self.enc2(x)
        enc_f3, x = self.enc3(x)
        enc_f4, _ = self.enc4(x)

        x = self.dec1(enc_f4, enc_f3)
        x = self.dec2(x, enc_f2)
        x = self.dec3(x, enc_f1)
        out = self.out_conv(x)
        return out



if __name__ == '__main__':
    from torch.autograd import Variable
    u = UNet()
    data = Variable(torch.randn(1, 112, 112, 80))
    out = u(data)
    print("out size: {}".format(out.size()))