import torch, os
from torch import nn
import torch.nn.functional as F
from torchvision import models
from huggingface_hub import PyTorchModelHubMixin


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        return input.contiguous().view(input.size(0), -1)


class Dense(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        activation="relu",
        kernel_initializer="he_normal",
    ):
        super(Dense, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.kernel_initializer = kernel_initializer

        self.linear = nn.Linear(in_features, out_features)
        # initialization
        if kernel_initializer == "he_normal":
            nn.init.kaiming_normal_(self.linear.weight)
        else:
            raise NotImplementedError

    def forward(self, inputs):
        outputs = self.linear(inputs)
        if self.activation is not None:
            if self.activation == "relu":
                outputs = nn.ReLU(inplace=True)(outputs)
        return outputs


class Conv2D(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, activation="relu", strides=1
    ):
        super(Conv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.activation = activation
        self.strides = strides

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, strides, int((kernel_size - 1) / 2)
        )
        # default: using he_normal as the kernel initializer
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, inputs):
        outputs = self.conv(inputs)
        if self.activation is not None:
            if self.activation == "relu":
                outputs = nn.ReLU(inplace=True)(outputs)
            else:
                raise NotImplementedError
        return outputs


class Decoder(nn.Module):
    def __init__(self, secret_size=100):
        super(Decoder, self).__init__()
        self.secret_size = secret_size
        # self.stn = SpatialTransformerNetwork()
        self.decoder = nn.Sequential(
            Conv2D(3, 32, 3, strides=2, activation="relu"),
            Conv2D(32, 32, 3, activation="relu"),
            Conv2D(32, 64, 3, strides=2, activation="relu"),
            Conv2D(64, 64, 3, activation="relu"),
            Conv2D(64, 64, 3, strides=2, activation="relu"),
            Conv2D(64, 128, 3, strides=2, activation="relu"),
            Conv2D(128, 128, 3, activation="relu"),
            Conv2D(128, 128, 3, strides=2, activation="relu"),
            Conv2D(128, 256, 3, strides=2, activation="relu"),
            Conv2D(256, 256, 3, activation="relu"),
            Conv2D(256, 256, 3, strides=2, activation="relu"),
            Conv2D(256, 512, 3, strides=2, activation="relu"),
            Conv2D(512, 512, 3, activation="relu"),
            Conv2D(512, 512, 3, strides=2, activation="relu"),
            Flatten(),
            Dense(512, 256, activation="relu"),
            Dense(256, secret_size, activation=None),
        )

    def forward(self, image):
        image = image - 0.5
        return torch.sigmoid(self.decoder(image))


class ConditionAdaptor(nn.Module):
    def __init__(self):
        super(ConditionAdaptor, self).__init__()

        self.secret_dense1 = Dense(100, 64 * 64, activation="relu")
        self.secret_dense2 = Dense(64 * 64, 3 * 64 * 64, activation="relu")
        self.conv1 = Conv2D(6, 6, 3, activation="relu")
        self.conv2 = Conv2D(6, 3, 3, activation=None)

    def forward(self, secrect, img_feature):
        secrect = 2 * (secrect - 0.5)
        secrect = self.secret_dense1(secrect)
        secrect = self.secret_dense2(secrect)
        secrect = secrect.reshape(-1, 3, 64, 64)

        secrect_enlarged = nn.Upsample(scale_factor=(4, 4))(secrect)
        inputs = torch.cat([secrect_enlarged, img_feature], dim=1)
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)

        return conv2


class ConditionAdaptor_orig(nn.Module):
    def __init__(self):
        super(ConditionAdaptor_orig, self).__init__()

        self.secret_dense1 = Dense(100, 64 * 64, activation="relu")
        self.secret_dense2 = Dense(64 * 64, 3 * 64 * 64, activation="relu")

        self.conv1 = Conv2D(6, 32, 3, activation="relu")
        self.conv2 = Conv2D(32, 32, 3, activation="relu", strides=2)
        self.conv3 = Conv2D(32, 64, 3, activation="relu", strides=2)
        self.conv4 = Conv2D(64, 128, 3, activation="relu", strides=2)
        self.conv5 = Conv2D(128, 256, 3, activation="relu", strides=2)
        self.up6 = Conv2D(256, 128, 3, activation="relu")
        self.conv6 = Conv2D(256, 128, 3, activation="relu")
        self.up7 = Conv2D(128, 64, 3, activation="relu")
        self.conv7 = Conv2D(128, 64, 3, activation="relu")
        self.up8 = Conv2D(64, 32, 3, activation="relu")
        self.conv8 = Conv2D(64, 32, 3, activation="relu")
        self.up9 = Conv2D(32, 32, 3, activation="relu")
        self.conv9 = Conv2D(70, 32, 3, activation="relu")
        self.conv10 = Conv2D(32, 32, 3, activation="relu")
        self.residual = Conv2D(32, 3, 1, activation=None)

    def forward(self, secrect, image):
        secrect = secrect - 0.5

        secrect = self.secret_dense1(secrect)
        secrect = self.secret_dense2(secrect)
        secrect = secrect.reshape(-1, 3, 64, 64)
        secrect_enlarged = nn.Upsample(scale_factor=(8, 8))(secrect)

        inputs = torch.cat([secrect_enlarged, image], dim=1)
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        up6 = self.up6(nn.Upsample(scale_factor=(2, 2))(conv5))
        merge6 = torch.cat([conv4, up6], dim=1)
        conv6 = self.conv6(merge6)
        up7 = self.up7(nn.Upsample(scale_factor=(2, 2))(conv6))
        merge7 = torch.cat([conv3, up7], dim=1)
        conv7 = self.conv7(merge7)
        up8 = self.up8(nn.Upsample(scale_factor=(2, 2))(conv7))
        merge8 = torch.cat([conv2, up8], dim=1)
        conv8 = self.conv8(merge8)
        up9 = self.up9(nn.Upsample(scale_factor=(2, 2))(conv8))
        merge9 = torch.cat([conv1, up9, inputs], dim=1)

        conv9 = self.conv9(merge9)
        conv10 = self.conv10(conv9)
        residual = self.residual(conv10)
        return residual


class CustomConvNeXt(nn.Module, PyTorchModelHubMixin):
    def __init__(self, secret_size, ckpt_path=None, device=None):
        super(CustomConvNeXt, self).__init__()
        self.convnext = models.convnext_base()
        self.convnext.classifier.append(
            nn.Linear(in_features=1000, out_features=secret_size, bias=True)
        )
        self.convnext.classifier.append(nn.Sigmoid())

        if ckpt_path is not None:
            self.load_ckpt_from_state_dict(ckpt_path, device)

    def load_ckpt_from_state_dict(self, ckpt_path, device):
        self.convnext.load_state_dict(
            torch.load(os.path.join(ckpt_path, "CustomConvNeXt.pth"))
        )
        self.convnext.to(device)

    def forward(self, x):
        x = self.convnext(x)
        return x
