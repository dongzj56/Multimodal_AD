"""
3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation
Paper URL: https://arxiv.org/abs/1606.06650
Author: Amir Aghdam
"""

from torch import nn
from torchsummary import summary
import torch
import time
import torch.nn.functional as F

# 编码端
class Conv3DBlock(nn.Module):
    """
    The basic block for double 3x3x3 convolutions in the analysis path
    -- __init__()
    :param in_channels -> number of input channels
    :param out_channels -> desired number of output channels
    :param bottleneck -> specifies the bottlneck block
    -- forward()
    :param input -> input Tensor to be convolved
    :return -> Tensor
    """

    def __init__(self, in_channels, out_channels, bottleneck = False) -> None:
        super(Conv3DBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels= in_channels, out_channels=out_channels//2, kernel_size=(3,3,3), padding=1)
        self.bn1 = nn.BatchNorm3d(num_features=out_channels//2)
        self.conv2 = nn.Conv3d(in_channels= out_channels//2, out_channels=out_channels, kernel_size=(3,3,3), padding=1)
        self.bn2 = nn.BatchNorm3d(num_features=out_channels)
        self.relu = nn.ReLU()
        self.bottleneck = bottleneck
        if not bottleneck:
            self.pooling = nn.MaxPool3d(kernel_size=(2,2,2), stride=2)

    
    def forward(self, input):
        res = self.relu(self.bn1(self.conv1(input)))
        res = self.relu(self.bn2(self.conv2(res)))
        out = None
        if not self.bottleneck:
            out = self.pooling(res)
        else:
            out = res
        return out, res



# 解码端
class UpConv3DBlock(nn.Module):
    """
    The basic block for upsampling followed by double 3x3x3 convolutions in the synthesis path
    -- __init__()
    :param in_channels -> number of input channels
    :param out_channels -> number of residual connections' channels to be concatenated
    :param last_layer -> specifies the last output layer
    :param num_classes -> specifies the number of output channels for dispirate classes
    -- forward()
    :param input -> input Tensor
    :param residual -> residual connection to be concatenated with input
    :return -> Tensor
    """

    def __init__(self, in_channels, res_channels=0, last_layer=False, num_classes=None) -> None:
        super(UpConv3DBlock, self).__init__()
        assert (last_layer==False and num_classes==None) or (last_layer==True and num_classes!=None), 'Invalid arguments'
        self.upconv1 = nn.ConvTranspose3d(in_channels=in_channels, out_channels=in_channels, kernel_size=(2, 2, 2), stride=2)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm3d(num_features=in_channels//2)
        self.conv1 = nn.Conv3d(in_channels=in_channels+res_channels, out_channels=in_channels//2, kernel_size=(3,3,3), padding=(1,1,1))
        self.conv2 = nn.Conv3d(in_channels=in_channels//2, out_channels=in_channels//2, kernel_size=(3,3,3), padding=(1,1,1))
        self.last_layer = last_layer
        if last_layer:
            self.conv3 = nn.Conv3d(in_channels=in_channels//2, out_channels=num_classes, kernel_size=(1,1,1))
            
        
    def forward(self, input, residual=None):
        out = self.upconv1(input)
        if residual!=None: out = torch.cat((out, residual), 1)
        out = self.relu(self.bn(self.conv1(out)))
        out = self.relu(self.bn(self.conv2(out)))
        if self.last_layer: out = self.conv3(out)
        return out
        

class UNet3D(nn.Module):
    """
    The basic block for upsampling followed by double 3x3x3 convolutions in the synthesis path
    -- __init__()
    :param in_channels -> number of input channels
    :param out_channels -> number of residual connections' channels to be concatenated
    :param last_layer -> specifies the last output layer
    :param num_classes -> specifies the number of output channels for dispirate classes
    -- forward()
    :param input -> input Tensor
    :param residual -> residual connection to be concatenated with input
    :return -> Tensor
    """
    # ---------------- 构造函数保持不变 ----------------
    def __init__(self, in_channels, num_classes,
                 level_channels=[64,128,256], bottleneck_channel=512):
        super().__init__()
        c1,c2,c3 = level_channels
        self.a_block1 = Conv3DBlock(in_channels, c1)
        self.a_block2 = Conv3DBlock(c1, c2)
        self.a_block3 = Conv3DBlock(c2, c3)
        self.bottleNeck = Conv3DBlock(c3, bottleneck_channel, bottleneck=True)

        self.s_block3 = UpConv3DBlock(bottleneck_channel, res_channels=c3)
        self.s_block2 = UpConv3DBlock(c3, res_channels=c2)
        self.s_block1 = UpConv3DBlock(c2, res_channels=c1,
                                      num_classes=num_classes, last_layer=True)

    # ----------- 辅助：pad→目标尺寸 & 记录 pad ---------
    @staticmethod
    def _pad_to_target(x, target=(96,112,96)):
        _, _, D,H,W = x.shape
        tD,tH,tW    = target
        pad = (0, tW-W,   0, tH-H,   0, tD-D)         # 只补右/后/下
        return F.pad(x, pad), pad                     # pad=(Wl,Wr,Hl,Hr,Dl,Dr)

    # ----------- 辅助：根据 pad 裁回原尺寸 ------------
    @staticmethod
    def _crop_back(y, pad):
        _, _, Dp,Hp,Wp = y.shape
        Dl,Dr = pad[4], pad[5]
        Hl,Hr = pad[2], pad[3]
        Wl,Wr = pad[0], pad[1]
        return y[:, :, Dl: Dp-Dr if Dr else None,
                        Hl: Hp-Hr if Hr else None,
                        Wl: Wp-Wr if Wr else None]

    # ------------------- forward --------------------
    def forward(self, x):
        # ① pad 到 96×112×96
        x_pad, pad = self._pad_to_target(x)

        # ---------- 编码 ----------
        out, res1 = self.a_block1(x_pad)
        out, res2 = self.a_block2(out)
        out, res3 = self.a_block3(out)
        out, _    = self.bottleNeck(out)

        # ---------- 解码 ----------
        out = self.s_block3(out, res3)
        out = self.s_block2(out, res2)
        out = self.s_block1(out, res1)

        # ② 裁回原始尺寸
        out = self._crop_back(out, pad)
        return out


# ------------------------- Demo -------------------------
if __name__ == '__main__':
    model = UNet3D(in_channels=3, num_classes=1)

    start_time = time.time()
    summary(model=model, input_size=(3, 16, 128, 128), batch_size=-1, device="cpu")
    print("--- %s seconds ---" % (time.time() - start_time))

    # 输入任意奇数尺寸 91×109×91
    dummy = torch.randn(1, 3, 91, 109, 91)
    with torch.no_grad():
        out = model(dummy)
    print("输入 :", dummy.shape)   # → torch.Size([1, 3, 91, 109, 91])
    print("输出 :", out.shape)     # → torch.Size([1, 1, 91, 109, 91])
