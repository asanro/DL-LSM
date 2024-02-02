""" Model architectures """
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from seed import seed_everything

class CNN2(nn.Module):
    def __init__(self, num_classes=1):
        super(CNN2, self).__init__()
        self.conv1 = nn.Conv3d(1, 50, kernel_size=10, stride=5)
        self.bn1 = nn.BatchNorm3d(50)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv3d(50, 100, kernel_size=10, stride=5)
        self.bn2 = nn.BatchNorm3d(100)
        self.relu2 = nn.ReLU()
        self.fc = nn.Linear(8000, num_classes)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# CNN2 ResNet
def conv3x3x3_(in_planes, out_planes, stride=1, dilation=1):
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=5,
        dilation=dilation,
        stride=stride,
        padding= 2, # 0
        bias=False
    )

def downsample_basic_block(x, planes, stride, no_cuda=False):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)
    ).zero_()
    if not no_cuda:
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

    out = torch.cat([out.data, zero_pads], dim=1)

    return out

class BasicBlock_(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(BasicBlock_, self).__init__()
        self.conv1 = conv3x3x3_(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3_(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
class ResidualCNN(nn.Module):
    def __init__(self, block, layers, num_classes=1, shortcut_type='B', no_cuda=False):
        self.inplanes = 50  # Update this based on your CNN2 model
        self.no_cuda = no_cuda
        super(ResidualCNN, self).__init__()
        self.conv1 = nn.Conv3d(
            1,
            50,  # Update this based on your CNN2 model
            kernel_size=10,
            stride=5,
            padding=0,
            bias=False)

        self.bn1 = nn.BatchNorm3d(50)  # Update this based on your CNN2 model
        self.relu = nn.ReLU(inplace=True)
          
        self.layer1 = self._make_layer(block, 50, layers[0], shortcut_type, stride=3)
        self.layer2 = self._make_layer(block, 100, layers[1], shortcut_type, stride=3)
        
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1)) # nn.AdaptiveAvgPool2d((1, 1)) 
        # Calculate fc_input_size dynamically
        self.fc_input_size = self.calculate_fc_input_size()
        self.fc = nn.Linear(self.fc_input_size, num_classes)
        
        # for parallel
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)  
                
    def calculate_fc_input_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, 152, 179, 142)
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.avgpool(x)
        return x.view(x.size(0), -1).shape[1]
    
    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    no_cuda=self.no_cuda)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1,
                              stride=stride, bias=False),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        
        x = self.avgpool(x3)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    
""" Other models """

class CNN2v2_skip(nn.Module):
    def __init__(self, num_classes=1):
        super(CNN2v2_skip, self).__init__()
        self.conv1 = nn.Conv3d(1, 35, kernel_size=10, stride=5)
        self.bn1 = nn.BatchNorm3d(35)
        self.relu1 = nn.ReLU()
        self.conv1x1 = nn.Conv3d(35, 75, kernel_size=1, stride=1)
    
        self.conv2 = nn.Conv3d(35, 75, kernel_size=10, stride=5)
        self.bn2 = nn.BatchNorm3d(75)
        self.relu2 = nn.ReLU()
        
        self.fc_input_size = self.calculate_fc_input_size()
        self.fc = nn.Linear(self.fc_input_size, num_classes)

    def calculate_fc_input_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, 152, 179, 142)
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            return x.view(x.size(0), -1).shape[1]

    def forward(self, x):
        x1 = self.relu1(self.bn1(self.conv1(x)))
        skip1 = self.conv1x1(x1)
        x2 = self.relu2(self.bn2(self.conv2(x1)))
        skip2 = x2
        skip1 = skip1[:, :, :x2.size(2), :x2.size(3), :x2.size(4)]

        x = x2 + skip1 + skip2
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        return x

class CNN3(nn.Module):
    def __init__(self, num_classes=1):
        super(CNN3, self).__init__()
        self.conv1 = nn.Conv3d(1, 25, kernel_size=10, stride=5) 
        self.bn1 = nn.BatchNorm3d(25)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv3d(25, 50, kernel_size=6, stride=3) # 5, 2
        self.bn2 = nn.BatchNorm3d(50)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv3d(50, 100, kernel_size=5, stride=2) # 10, 5
        self.bn3 = nn.BatchNorm3d(100)
        self.relu3 = nn.ReLU()
        self.fc_input_size = self.calculate_fc_input_size()
        self.fc = nn.Linear(self.fc_input_size, num_classes)

    def calculate_fc_input_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, 152, 179, 142)  
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            return x.view(x.size(0), -1).shape[1]
        
    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    

class CNN2_Pool(nn.Module): # looses information
    def __init__(self, num_classes=1):
        super(CNN2_Pool, self).__init__()
        self.conv1 = nn.Conv3d(1, 100, kernel_size=10, stride=5)
        self.bn1 = nn.BatchNorm3d(100)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv3d(100, 200, kernel_size=10, stride=5)
        self.bn2 = nn.BatchNorm3d(200)
        self.relu2 = nn.ReLU()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=1)  # when pool 1600
        self.fc = nn.Linear(7200, num_classes)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        # print(x.size())
        x = self.fc(x)
        return x
    
class DilatedCNN2(nn.Module): # out of memory
    def __init__(self, num_classes=1):
        super(DilatedCNN2, self).__init__()
        
        # Define dilated convolutions
        self.conv1 = nn.Conv3d(1, 75, kernel_size=10, stride=5, padding=1, dilation=1)
        self.conv2 = nn.Conv3d(75, 150, kernel_size=10, stride=5, padding=2, dilation=2)
        
        # Batch normalization and ReLU
        self.bn1 = nn.BatchNorm3d(75)
        self.bn2 = nn.BatchNorm3d(150)
        self.relu = nn.ReLU()
        
        # Global average pooling
        # self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        
        self.fc_input_size = self.calculate_fc_input_size()
        self.fc = nn.Linear(self.fc_input_size, num_classes)

    def calculate_fc_input_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, 152, 179, 142)  
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.relu(self.bn2(self.conv2(x)))
            return x.view(x.size(0), -1).shape[1]
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        
        # Global average pooling
        # x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        
        # Fully connected layer
        x = self.fc(x)
        
        return x

class DilatedCNN3(nn.Module):
    def __init__(self, num_classes=1):
        super(DilatedCNN3, self).__init__()
        
        # Define dilated convolutions
        self.conv1 = nn.Conv3d(1, 25, kernel_size=10, stride=5, padding=1, dilation=1)
        self.conv2 = nn.Conv3d(25, 50, kernel_size=6, stride=3, padding=2, dilation=2)
        self.conv3 = nn.Conv3d(50, 100, kernel_size=5, stride=2, padding=3, dilation=3)
        
        # Batch normalization and ReLU
        self.bn1 = nn.BatchNorm3d(25)
        self.bn2 = nn.BatchNorm3d(50)
        self.bn3 = nn.BatchNorm3d(100)
        self.relu = nn.ReLU()
        
        # Global average pooling
        # self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        
        self.fc_input_size = self.calculate_fc_input_size()
        self.fc = nn.Linear(self.fc_input_size, num_classes)

    def calculate_fc_input_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, 152, 179, 142)  
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.relu(self.bn2(self.conv2(x)))
            x = self.relu(self.bn3(self.conv3(x)))
            return x.view(x.size(0), -1).shape[1]
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        
        # Global average pooling
        # x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        
        # Fully connected layer
        x = self.fc(x)
        
        return x
    
class CNN4_16(nn.Module):
    def __init__(self, num_classes=1):
        super(CNN4_16, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=(3, 4, 4), stride=1, padding=0)
        self.bn1 = nn.BatchNorm3d(16)
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)

        self.conv2 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=1, padding=0)
        self.bn2 = nn.BatchNorm3d(32)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)

        self.conv3 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=1, padding=0)
        self.bn3 = nn.BatchNorm3d(64)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)

        self.conv4 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=1, padding=0)
        self.bn4 = nn.BatchNorm3d(128)
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)

        self.flatten = nn.Flatten()
        # Adjust input size based on output from conv layers
        self.fc1 = nn.Linear(128 * 7 * 9 * 6, 256)
        self.fc2 = nn.Linear(256, num_classes)
        # self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        x = self.pool4(torch.relu(self.bn4(self.conv4(x))))

        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        # x = self.dropout(x)
        x = self.fc2(x)
        return x

    
# RESNET10
def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        dilation=dilation,
        stride=stride,
        padding=dilation,
        bias=False
    )


def downsample_basic_block(x, planes, stride, no_cuda=False):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)
    ).zero_()
    if not no_cuda:
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

    out = torch.cat([out.data, zero_pads], dim=1)

    return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
    
class ResNet2(nn.Module):
    def __init__(self,
                 block,
                 layers,
                 num_classes=3,
                 shortcut_type='B',
                 no_cuda = False):
        self.inplanes = 64
        self.no_cuda = no_cuda
        super(ResNet2, self).__init__()
        self.conv1 = nn.Conv3d(
            1,
            64,
            kernel_size=7,
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False)
            
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=1, dilation=4)
        
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
    
        # Calculate fc_input_size dynamically
        self.fc_input_size = self.calculate_fc_input_size()
        self.fc = nn.Linear(self.fc_input_size, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)  
                
    def calculate_fc_input_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, 152, 179, 142)
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)
        return x.view(x.size(0), -1).shape[1]
    
    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                # downsample
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    no_cuda=self.no_cuda)
            else:
                # conv 1x1
                downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, 
                              stride=stride, bias=False), 
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
     
        
def conv3x3x3_7(in_planes, out_planes, stride=1, dilation=1):
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=7,
        dilation=dilation,
        stride=stride,
        padding= 3,
        bias=False
    )


class BasicBlock_7(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(BasicBlock_7, self).__init__()
        self.conv1 = conv3x3x3_7(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3_7(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
    
class CNN2ResNet_maxpool(nn.Module):
    def __init__(self, block, layers, num_classes=1, shortcut_type='B', no_cuda=False):
        self.inplanes = 50  # Update this based on your CNN2 model
        self.no_cuda = no_cuda
        super(CNN2ResNet_maxpool, self).__init__()
        self.conv1 = nn.Conv3d(
            1,
            50,  # Update this based on your CNN2 model
            kernel_size=10,
            stride=5,
            padding=0,
            bias=False)

        self.bn1 = nn.BatchNorm3d(50)  # Update this based on your CNN2 model
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)  
        self.layer1 = self._make_layer(block, 50, layers[0], shortcut_type, stride=3)
        self.layer2 = self._make_layer(block, 100, layers[1], shortcut_type, stride=3)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Calculate fc_input_size dynamically
        self.fc_input_size = self.calculate_fc_input_size()
        self.fc = nn.Linear(self.fc_input_size, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)  
                
    def calculate_fc_input_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, 152, 179, 142)
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.avgpool(x)
        return x.view(x.size(0), -1).shape[1]
    
    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    no_cuda=self.no_cuda)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1,
                              stride=stride, bias=False),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
        
class ResNet_andrea(nn.Module):
    def __init__(self, block, layers, num_classes=3, shortcut_type='B', no_cuda = False):
        self.inplanes = 64
        self.no_cuda = no_cuda
        super(ResNet_andrea, self).__init__()
        self.conv1 = nn.Conv3d(
            1,
            64,
            kernel_size=7,
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False)
            
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=1, dilation=4)
        
        self.conv_seg = nn.Sequential(
            nn.ConvTranspose3d(512 * block.expansion,32,2,stride=2),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32,32,kernel_size=3,stride=(1, 1, 1),padding=(1, 1, 1),bias=False), 
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32,8,kernel_size=1,stride=(1, 1, 1),bias=False)
        )

        # Calculate fc_input_size dynamically
        self.fc_input_size = self.calculate_fc_input_size()
        self.fc = nn.Linear(self.fc_input_size, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)   
    def calculate_fc_input_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, 152, 179, 142)
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.conv_seg(x)
        return x.view(x.size(0), -1).shape[1]
    
    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    no_cuda=self.no_cuda)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv_seg(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    
# Other deep models
class CNN2_32(nn.Module):  ### Probar el nuevo
    def __init__(self, num_classes=1):
        super(CNN2_32, self).__init__()
        self.conv1 = nn.Conv3d(1, 60, kernel_size=5, stride=2, padding=1)
        self.bn1 = nn.BatchNorm3d(60)
        self.pool1 = nn.MaxPool3d(kernel_size=3, stride=2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv3d(60, 120, kernel_size=5, stride=2, padding=1)
        self.bn2 = nn.BatchNorm3d(120)
        self.pool2 = nn.MaxPool3d(kernel_size=3, stride=2)
        self.relu2 = nn.ReLU()

        # Calculate the correct input size for the fully connected layer
        self.fc_input_size = self._calculate_fc_input_size()
        self.fc = nn.Linear(self.fc_input_size, num_classes)

    def _calculate_fc_input_size(self):
        # Define a forward pass to calculate the input size of the fully connected layer
        with torch.no_grad():
            x = torch.zeros(1, 1, 152, 179, 142)  # Create a dummy input with the correct size
            x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
            x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
            return x.view(x.size(0), -1).shape[1]

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        return x


# Ryanne's model
class conv2_relu(nn.Module):
    def __init__(self, num_classes=1):
        super(conv2_relu, self).__init__()
        self.conv1 = nn.Conv3d(1, 75, kernel_size=10, stride=5)
        self.bn1 = nn.BatchNorm3d(75)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv3d(75, 150, kernel_size=10, stride=5)
        self.bn2 = nn.BatchNorm3d(150)
        self.relu2 = nn.ReLU()
        self.fc = nn.Linear(12000, num_classes)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

