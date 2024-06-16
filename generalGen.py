import torch
import torch.nn as nn
import torchvision.transforms.functional as F

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False):
        super().__init__()

        if down:
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False, padding_mode="reflect"
            )
        else:
            self.conv = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False
            )
        
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return self.dropout(x) if self.use_dropout else x

class Generator(nn.Module):
    def __init__(self, in_channels, features=64):
        super().__init__()

        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )

        self.down1 = Block(features, features * 2, act="leaky")
        self.down2 = Block(features * 2, features * 4, act="leaky")
        self.down3 = Block(features * 4, features * 8, act="leaky")
        self.down4 = Block(features * 8, features * 8, act="leaky")
        self.down5 = Block(features * 8, features * 8, act="leaky")
        self.down6 = Block(features * 8, features * 8, act="leaky")

        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )

        self.up1 = Block(features * 8, features * 8, down=False, act="relu", use_dropout=True)
        self.up2 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True)
        self.up3 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True)
        self.up4 = Block(features * 8 * 2, features * 8, down=False, act="relu")
        self.up5 = Block(features * 8 * 2, features * 4, down=False, act="relu")
        self.up6 = Block(features * 4 * 2, features * 2, down=False, act="relu")
        self.up7 = Block(features * 2 * 2, features, down=False, act="relu")

        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features * 2, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        if d7.shape[2:] != (2,2):
            d7 = F.resize(d7 , size = (2,2) , antialias=True)
        bottleneck = self.bottleneck(d7)
        
        # print("d1: " , d1.shape)
        # print("d2: " , d2.shape)
        # print("d3: " , d3.shape)
        # print("d4: " , d4.shape)
        # print("d5: " , d5.shape)
        # print("d6: " , d6.shape)
        # print("d7: " , d7.shape)
        # print("bottleneck " , bottleneck.shape)


        up1 = self.up1(bottleneck)
        
        if up1.shape != d7.shape:
            up1 = F.resize(up1 , size = d7.shape[2:] , antialias=True)
        up2 = self.up2(torch.cat([up1, d7], dim=1))
        if up2.shape != d6.shape:
            up2 = F.resize(up1 , size = d6.shape[2:] , antialias=True)
        up3 = self.up3(torch.cat([up2, d6], dim=1))
        if up3.shape != d5.shape:
            up3 = F.resize(up3 , size = d5.shape[2:] , antialias=True)
        up4 = self.up4(torch.cat([up3, d5], dim=1))
        if up4.shape != d4.shape:
            up4 = F.resize(up4 , size = d4.shape[2:] , antialias=True)
        up5 = self.up5(torch.cat([up4, d4], dim=1))
        if up5.shape != d3.shape:
            up5 = F.resize(up5 , size = d3.shape[2:] , antialias=True)
        up6 = self.up6(torch.cat([up5, d3], dim=1))
        if up6.shape != d2.shape:
            up6 = F.resize(up6 , size = d2.shape[2:] , antialias=True)
        up7 = self.up7(torch.cat([up6, d2], dim=1))
        if up7.shape != d1.shape:
            up7 = F.resize(up7 , size = d1.shape[2:] , antialias=True)
        
        # print("up1: " , up1.shape)
        # print("up2: " , up2.shape)
        # print("up3: " , up3.shape)
        # print("up4: " , up4.shape)
        # print("up5: " , up5.shape)
        # print("up6: " , up6.shape)
        # print("up7: " , up7.shape)
        
        return self.final_up(torch.cat([up7, d1], dim=1))

if __name__ == "__main__":
    x = torch.randn((1, 3, 152, 257))
    model = Generator(3)
    pred = model(x)
    print(pred.shape)

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# def pad_to(x, stride):
#     h, w = x.shape[-2:]

#     if h % stride > 0:
#         new_h = h + stride - h % stride
#     else:
#         new_h = h
#     if w % stride > 0:
#         new_w = w + stride - w % stride
#     else:
#         new_w = w
#     lh, uh = int((new_h-h) / 2), int(new_h-h) - int((new_h-h) / 2)
#     lw, uw = int((new_w-w) / 2), int(new_w-w) - int((new_w-w) / 2)
#     pads = (lw, uw, lh, uh)

#     out = F.pad(x, pads, "constant", 0)

#     return out, pads

# def unpad(x, pad):
#     if pad[2]+pad[3] > 0:
#         x = x[:,:,pad[2]:-pad[3],:]
#     if pad[0]+pad[1] > 0:
#         x = x[:,:,:,pad[0]:-pad[1]]
#     return x

# class Block(nn.Module):
#     def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False):
#         super().__init__()

#         if down:
#             self.conv = nn.Conv2d(
#                 in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False, padding_mode="reflect"
#             )
#         else:
#             self.conv = nn.ConvTranspose2d(
#                 in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False
#             )
        
#         self.norm = nn.BatchNorm2d(out_channels)
#         self.act = nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2)
#         self.use_dropout = use_dropout
#         self.dropout = nn.Dropout(0.5)
        
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.norm(x)
#         x = self.act(x)
#         return self.dropout(x) if self.use_dropout else x

# class Generator(nn.Module):
#     def __init__(self, in_channels, features=64):
#         super().__init__()

#         self.initial_down = nn.Sequential(
#             nn.Conv2d(in_channels, features, kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
#             nn.LeakyReLU(0.2)
#         )

#         self.down1 = Block(features, features * 2, act="leaky")
#         self.down2 = Block(features * 2, features * 4, act="leaky")
#         self.down3 = Block(features * 4, features * 8, act="leaky")
#         self.down4 = Block(features * 8, features * 8, act="leaky")
#         self.down5 = Block(features * 8, features * 8, act="leaky")
#         self.down6 = Block(features * 8, features * 8, act="leaky")

#         self.bottleneck = nn.Sequential(
#             nn.Conv2d(features * 8, features * 8, kernel_size=4, stride=2, padding=1),
#             nn.ReLU()
#         )

#         self.up1 = Block(features * 8, features * 8, down=False, act="relu", use_dropout=True)
#         self.up2 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True)
#         self.up3 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True)
#         self.up4 = Block(features * 8 * 2, features * 8, down=False, act="relu")
#         self.up5 = Block(features * 8 * 2, features * 4, down=False, act="relu")
#         self.up6 = Block(features * 4 * 2, features * 2, down=False, act="relu")
#         self.up7 = Block(features * 2 * 2, features, down=False, act="relu")

#         self.final_up = nn.Sequential(
#             nn.ConvTranspose2d(features * 2, in_channels, kernel_size=4, stride=2, padding=1),
#             nn.Tanh()
#         )

#     def forward(self, x):
#         padded_x, pads = pad_to(x, stride=64)

#         d1 = self.initial_down(padded_x)
#         d2 = self.down1(d1)
#         d3 = self.down2(d2)
#         d4 = self.down3(d3)
#         d5 = self.down4(d4)
#         d6 = self.down5(d5)
#         d7 = self.down6(d6)

#         if d7.size(2) < 4 or d7.size(3) < 4:
#             d7 = F.interpolate(d7, size=(max(4, d7.size(2)), max(4, d7.size(3))), mode='nearest')

#         bottleneck = self.bottleneck(d7)

#         up1 = self.up1(bottleneck)
#         up1 = F.interpolate(up1, size=(d7.size(2), d7.size(3)), mode='nearest')
#         up2 = self.up2(torch.cat([up1, d7], dim=1))
#         up2 = F.interpolate(up2, size=(d6.size(2), d6.size(3)), mode='nearest')
#         up3 = self.up3(torch.cat([up2, d6], dim=1))
#         up3 = F.interpolate(up3, size=(d5.size(2), d5.size(3)), mode='nearest')
#         up4 = self.up4(torch.cat([up3, d5], dim=1))
#         up4 = F.interpolate(up4, size=(d4.size(2), d4.size(3)), mode='nearest')
#         up5 = self.up5(torch.cat([up4, d4], dim=1))
#         up5 = F.interpolate(up5, size=(d3.size(2), d3.size(3)), mode='nearest')
#         up6 = self.up6(torch.cat([up5, d3], dim=1))
#         up6 = F.interpolate(up6, size=(d2.size(2), d2.size(3)), mode='nearest')
#         up7 = self.up7(torch.cat([up6, d2], dim=1))
#         up7 = F.interpolate(up7, size=(d1.size(2), d1.size(3)), mode='nearest')

#         final_output = self.final_up(torch.cat([up7, d1], dim=1))
#         final_output = unpad(final_output, pads)

#         return final_output

# if __name__ == "__main__":
#     x = torch.randn((1, 3, 152, 257))
#     model = Generator(3)
#     pred = model(x)
#     print(pred.shape)
