import torch
import torchvision
import torch.nn as nn

F = nn.functional

class SegNet(nn.Module):
    def __init__(self):
        super(SegNet, self).__init__()
        #encoder
        # encoder block1
        self.enconv_11 = nn.Sequential(nn.Conv2d(3,3,3, padding = 1), nn.BatchNorm2d(3))
        self.encoder_conv1 = nn.Conv2d(3, 32, 3, padding = 1)
        self.encoder_bn1 = nn.BatchNorm2d(32)
        self.encoder_max1 = nn.MaxPool2d(2, 2, return_indices = True)
        #encoder block2
        self.enconv_21 = nn.Sequential(nn.Conv2d(32,32,3, padding = 1), nn.BatchNorm2d(32))
        self.encoder_conv2 = nn.Conv2d(32, 64, 3, padding = 1)
        self.encoder_bn2 = nn.BatchNorm2d(64)
        self.encoder_max2 = nn.MaxPool2d(2, 2, return_indices = True)
        #encoder block3
        self.enconv_31 = nn.Sequential(nn.Conv2d(64,64,3, padding = 1),nn.BatchNorm2d(64)) 
        self.encoder_conv3 = nn.Conv2d(64, 128, 3, padding = 1)
        self.encoder_bn3 = nn.BatchNorm2d(128)
        self.encoder_max3 = nn.MaxPool2d(2, 2, return_indices = True)
        #encoder block4
        self.enconv_41 = nn.Sequential(nn.Conv2d(128,128, 3, padding = 1), nn.BatchNorm2d(128))
        self.encoder_conv4 = nn.Conv2d(128, 256, 3, padding = 1)
        self.encoder_bn4 = nn.BatchNorm2d(256)
        self.encoder_max4 = nn.MaxPool2d(2, 2, return_indices = True)
        #encoder block5
        self.enconv_51 = nn.Sequential(nn.Conv2d(256,256,3, padding = 1), nn.BatchNorm2d(256))
        self.encoder_conv5 = nn.Conv2d(256, 512, 5, padding = 1)
        self.encoder_bn5 = nn.BatchNorm2d(512)
        self.encoder_max5 = nn.MaxPool2d(2, 2, return_indices = True, padding = 1)
        self.bottleneck = nn.Conv2d(512,512,1)
        self.bottleneck_bn = nn.BatchNorm2d(512)
        #decoder block1
        self.decoder_unpool1 = nn.MaxUnpool2d(2)
        self.decoder_conv1 = nn.Conv2d(512, 256 , 1, padding = 0)
        self.decoder_bn1 = nn.BatchNorm2d(256)
        self.deconv_11 = nn.Sequential(nn.Conv2d(256,256,3, padding  =1), nn.BatchNorm2d(256))
        #decoder block2
        self.decoder_unpool2 = nn.MaxUnpool2d(2)
        self.decoder_conv2 = nn.Conv2d(256, 128 , (4,3), padding = (2,1))
        self.decoder_bn2 = nn.BatchNorm2d(128)
        self.deconv_21 = nn.Sequential(nn.Conv2d(128,128,3, padding = 1), nn.BatchNorm2d(128))
        #decoder block3
        self.decoder_unpool3 = nn.MaxUnpool2d(2)
        self.decoder_conv3 = nn.Conv2d(128, 64 , 1, padding = 0)
        self.decoder_bn3 = nn.BatchNorm2d(64)
        self.deconv_31 = nn.Sequential(nn.Conv2d(64,64,3, padding = 1), nn.BatchNorm2d(64))
        #decoder block4
        self.decoder_unpool4 = nn.MaxUnpool2d(2)
        self.decoder_conv4 = nn.Conv2d(64, 32 , 3, padding = 1)
        self.decoder_bn4 = nn.BatchNorm2d(32)
        self.deconv_41 = nn.Sequential(nn.Conv2d(32, 32, 3, padding = 1), nn.BatchNorm2d(32))
        #decoder block5
        self.decoder_unpool5 = nn.MaxUnpool2d(2)
        self.decoder_conv5 = nn.Conv2d(32, 3 , 3, padding = 1)
        self.decoder_bn5 = nn.BatchNorm2d(3)
        self.deconv_51 = nn.Sequential(nn.Conv2d(3,3,3, padding = 1), nn.BatchNorm2d(3))
        self.classifier = nn.Conv2d(3,8, 1)
    
    def forward(self, x):
        #encoder
        x = F.relu(self.enconv_11(x))
        x, index1 = self.encoder_max1(F.relu(self.encoder_bn1(self.encoder_conv1(x))))
        #print("conv1: {}".format(x.shape))
        x = F.relu(self.enconv_21(x))
        x, index2 = self.encoder_max2(F.relu(self.encoder_bn2(self.encoder_conv2(x))))
        #print("conv2: {}".format(x.shape))
        x = F.relu(self.enconv_31(x))
        x, index3 = self.encoder_max3(F.relu(self.encoder_bn3(self.encoder_conv3(x))))
        #print("conv3: {}".format(x.shape))
        x = F.relu(self.enconv_41(x))
        x, index4 = self.encoder_max4(F.relu(self.encoder_bn4(self.encoder_conv4(x))))
        #print("conv4: {}".format(x.shape))
        x = F.relu(self.enconv_51(x))
        x, index5 = self.encoder_max5(F.relu(self.encoder_bn5(self.encoder_conv5(x))))
        #print("conv5: {}".format(x.shape))
        #bottleneck layer
        x = F.relu(self.bottleneck_bn(self.bottleneck(x)))
        #decoder
        x = F.relu(self.decoder_bn1(self.decoder_conv1(self.decoder_unpool1(x,index5))))
        x = F.relu(self.deconv_11(x))
        #print("deconv1: {}".format(x.shape))
        x = F.relu(self.decoder_bn2(self.decoder_conv2(self.decoder_unpool2(x,index4))))
        x = F.relu(self.deconv_21(x))
        #print("deconv2: {}".format(x.shape))
        x = F.relu(self.decoder_bn3(self.decoder_conv3(self.decoder_unpool3(x,index3))))
        x = F.relu(self.deconv_31(x))
        #print("deconv3: {}".format(x.shape))
        x = F.relu(self.decoder_bn4(self.decoder_conv4(self.decoder_unpool4(x,index2))))
        x = F.relu(self.deconv_41(x))
        #print("deconv4: {}".format(x.shape))
        x = F.relu(self.decoder_bn5(self.decoder_conv5(self.decoder_unpool5(x,index1))))
        x = F.relu(self.deconv_51(x))
        x = self.classifier(x)
        return x
        return x