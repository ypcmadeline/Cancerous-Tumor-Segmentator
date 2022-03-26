from torch import nn
import torch
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, channel):
        super(Encoder, self).__init__()
        self.enc = nn.Sequential(
            nn.Conv3d(channel, channel, 3, stride=1, padding=1),
            # nn.BatchNorm3d(channel),
            nn.PReLU(),
            nn.Conv3d(channel, channel, 3, stride=1, padding=1),
            # nn.BatchNorm3d(channel),
            nn.PReLU(),
            nn.Conv3d(channel, channel, 3, stride=1, padding=1),
            # nn.BatchNorm3d(channel),
            nn.PReLU()
        )
    
    def forward(self, x):
        return self.enc(x)


class Decoder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Decoder, self).__init__()
        self.in_ch = in_ch
        self.dec_blocks = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, 1, padding=1),
            nn.PReLU(out_ch),

            nn.Conv3d(out_ch, out_ch, 3, 1, padding=1),
            nn.PReLU(out_ch),

            nn.Conv3d(out_ch, out_ch, 3, 1, padding=1),
            nn.PReLU(out_ch),
        )

        
    def forward(self, x):
        return self.dec_blocks(x)


class DownConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DownConv, self).__init__()
        self.down_conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 2, 2),
            nn.PReLU(out_ch)
        )
        
    def forward(self, x):
        return self.down_conv(x)

class UpConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpConv, self).__init__()
        self.up_conv = nn.Sequential(
            nn.ConvTranspose3d(in_ch, out_ch, 2, 2),
            nn.PReLU(out_ch)
        )
        
    def forward(self, x):
        return self.up_conv(x)

class Map(nn.Module):
    def __init__(self, ch, factor):
        super(Map, self).__init__()
        self.map = nn.Sequential(
            nn.Conv3d(ch, 2, 1, 1),
            nn.Upsample(scale_factor=factor, mode='trilinear', align_corners=False),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        return self.map(x)



class UNet(nn.Module):
    def __init__(self, training=True):
        super(UNet, self).__init__()
        self.training = training
        self.drop = 0.2
        self.enc1 = nn.Sequential(
                        nn.Conv3d(1, 16, 3, 1, padding=1),
                        nn.BatchNorm3d(16),
                        nn.PReLU(16),
                        nn.Conv3d(16, 16, 3, 1, padding=1),
                        nn.BatchNorm3d(16),
                        nn.PReLU(16),
                    )
        self.enc2 = Encoder(32)
        self.enc3 = Encoder(64)
        self.enc4 = Encoder(128)

        self.dec1 = Decoder(128, 256)
        self.dec2 = Decoder(128+64, 128)
        self.dec3 = Decoder(64+32, 64)
        self.dec4 = nn.Sequential(
                        nn.Conv3d(32+16, 32, 3, 1, padding=1),
                        nn.PReLU(32),
                        nn.Conv3d(32, 32, 3, 1, padding=1),
                        nn.PReLU(32),
                    )

        self.dn_conv1 = DownConv(16, 32)
        self.dn_conv2 = DownConv(32, 64)
        self.dn_conv3 = DownConv(64, 128)
        self.dn_conv4 = nn.Sequential(
                            nn.Conv3d(128, 256, 3, 1, padding=1),
                            nn.PReLU(256)
                        )

        self.up_conv2 = UpConv(256, 128)
        self.up_conv3 = UpConv(128, 64)
        self.up_conv4 = UpConv(64, 32)

        self.map4 = Map(32, (1,1,1))
        self.map3 = Map(64, (2,2,2))
        self.map2 = Map(128, (4,4,4))
        self.map1 = Map(256, (8,8,8))


    def forward(self, x):

        long1 = self.enc1(x) + x
        sc1 = self.dn_conv1(long1)

        long2 = self.enc2(sc1) + sc1
        long2 = F.dropout(long2, self.drop, self.training)
        sc2 = self.dn_conv2(long2)

        long3 = self.enc3(sc2) + sc2
        long3 = F.dropout(long3, self.drop, self.training)
        sc3 = self.dn_conv3(long3)

        long4 = self.enc4(sc3) + sc3
        long4 = F.dropout(long4, self.drop, self.training)
        sc4 = self.dn_conv4(long4)

        out = self.dec1(long4) + sc4
        out = F.dropout(out, self.drop, self.training)

        out1 = self.map1(out)
        sc6 = self.up_conv2(out)

        out = self.dec2(torch.cat([sc6, long3], dim=1)) + sc6
        out = F.dropout(out, self.drop, self.training)

        out2 = self.map2(out)
        sc7 = self.up_conv3(out)

        out = self.dec3(torch.cat([sc7, long2], dim=1)) + sc7
        out = F.dropout(out, self.drop, self.training)

        out3 = self.map3(out)
        sc8 = self.up_conv4(out)

        out = self.dec4(torch.cat([sc8, long1], dim=1)) + sc8
        out4 = self.map4(out)

        if self.training:
            return out1, out2, out3, out4
        else:
            return out4


if __name__ == '__main__':
    from torch.autograd import Variable
    u = UNet(training=True)
    data = Variable(torch.randn(1, 1, 112, 112, 80))
    out = u(data)
    # print(u)
    print(out[3].shape)