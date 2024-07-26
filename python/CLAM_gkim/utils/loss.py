import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function, Variable
import numpy as np
from math import exp

# https://becominghuman.ai/investigating-focal-and-dice-loss-for-the-kaggle-2018-data-science-bowl-65fb9af4f36c
class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma
        
    def forward(self, input, target):
        # Inspired by the implementation of binary_cross_entropy_with_logits
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

        # This formula gives us the log sigmoid of 1-p if y is 0 and of p if y is 1
        invprobs = F.logsigmoid(-input * (target * 2 - 1))
        loss = (invprobs * self.gamma).exp() * loss
        
        return loss.mean()

class TverskyLoss:
    def __init__(self, alpha, torch_device):
        # super(TverskyLoss, self).__init__()
        self.a = alpha
        self.b = 1 - alpha 
        self.smooth = torch.tensor(1.0, device=torch_device)

    def __call__(self, predict, target_):
        predict = F.sigmoid(predict)
        target_f  = target_.view(-1) # g
        predict_f = predict.view(-1) # p

        # PG + a * P_G + b * G_P        
        PG  = (predict_f * target_f).sum() # p0g0
        P_G = (predict_f * (1 - target_f)).sum() * self.a # p0g1
        G_P = ((1 - predict_f) * target_f).sum() * self.b # p1g0

        loss = PG / (PG + P_G + G_P + self.smooth)
        return loss * -1


def dice_coeff_loss(prob, label, nlabels=1):
    '''
    automl/agent.py#L16
    '''
    max_val, pred = prob.max(1)
    fg_mask = max_val.gt(0.5).type_as(label)

    # masking
    dices_per_label = []
    smooth = 1e-10
    eps = 1e-8
    for l in range(0, nlabels):
        dices = []
        for n in range(prob.size(0)):
            label_p = label[n].eq(l).float()

            if l == 0:
                prob_l = 1-max_val[n]
            else:
                prob_l = prob[n, l-1, :, :, :]
            prob_l = torch.clamp(prob_l, eps, 1.0-eps)
            # calc accuracy
            jacc = 1.0 - torch.clamp(( ((prob_l * label_p).sum() + smooth) / ((label_p**2).sum() + (prob_l**2).sum() - (prob_l * label_p).sum() + smooth) ), 0.0, 1.0)
            dices.append(jacc.view(-1))

        dices_per_label.append(torch.mean(torch.cat(dices)).view(-1))

    return torch.mean(torch.cat(dices_per_label))


# https://gist.github.com/weiliu620/52d140b22685cf9552da4899e2160183
def dice_loss(pred, target):
    """
    This definition generalize to real valued pred and target vector.
    This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """
    pred = F.sigmoid(pred)
    smooth = 1.

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(tflat * iflat)
    B_sum = torch.sum(tflat * tflat)

    return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth) )


def bnn_regression_loss(output, target, l1=True):
    mean, log_var = output[:, 0], output[:, 1]
    precision = (-log_var).exp()
    if l1:
        # return (precision.sqrt() * (target - mean).abs() + 0.5 * log_var).mean()
        return torch.nn.MSELoss()(target, mean)
    else:
        return (0.1 * precision * (target - mean).pow(2) + log_var).mean()
    # return nn.MSELoss()(mean, target)
    # mse = nn.MSELoss(reduction='none')
    # return precision.mul(mse(mean, target)).add(0.5 * log_var).mean()
    # return target.sub(mean).abs().div(precision).mean()
    # return target.sub(mean).pow(2).div(log_var.exp()).add(log_var).mean()


class Sobel3d(torch.nn.Module):
    def __init__(self, channel_out):
        super(Sobel3d, self).__init__()
        gz, gy, gx = [torch.zeros([3, 3, 3]) for i in range(3)]
        hx = hy = hz = torch.Tensor([1, 2, 1])
        hpx = hpy = hpz = torch.Tensor([1, 0, -1])

        for m in range(3):
            for n in range(3):
                for k in range(3):
                    gz[m, n, k] = hpz[m] * hy[n] * hx[k]
                    gy[m, n, k] = hz[m] * hpy[n] * hx[k]
                    gx[m, n, k] = hz[m] * hy[n] * hpx[k]

        gz = gz.unsqueeze(dim = 0).unsqueeze(dim = 0)
        gy = gy.unsqueeze(dim = 0).unsqueeze(dim = 0)
        gx = gx.unsqueeze(dim = 0).unsqueeze(dim = 0)
        #self.kernel = [gz gy gx][axis] # Geon: idk why the axis was specified before
        self.kernel = torch.cat((gz, gy, gx), axis = 0)
        self.kernel = self.kernel.repeat([1,channel_out,1,1,1])
        self.sobel = torch.nn.Conv3d(in_channels=self.kernel.shape[1], out_channels=self.kernel.shape[0], kernel_size=3, padding=1, bias=False).to(torch.cuda.current_device())
        self.sobel.weight.data.copy_(self.kernel)
        self.sobel.weight.requires_grad = False

    def forward(self, img):
        return self.sobel(img)

class Sobel2d(torch.nn.Module):
    def __init__(self):#, channel_out): # output channel of the whole network (ex: depth)
        super(Sobel2d, self).__init__()

        kernely = torch.Tensor([[-1., 0., 1.],[-2., 0., 2.],[-1., 0., 1.]]).unsqueeze(dim = 0).unsqueeze(dim = 0)
        kernelx = torch.Tensor([[-1., -2., -1.],[0., 0., 0.],[1., 2., 1.]]).unsqueeze(dim = 0).unsqueeze(dim = 0)
        self.kernel = torch.cat((kernelx,kernely), axis = 0)

        # forward ?먯꽌 concatenation ?섎뒗 寃??몄뿉???대떦 channel ?쒖쇅?섍퀬 0??kernel ?щ윭 媛?留뚮뱾 ???덉쑝??        # memory 議곌툑?대씪???꾨겮湲??꾪빐 forward ?먯꽌 concat
        self.sobel = torch.nn.Conv2d(in_channels=self.kernel.shape[1], out_channels=self.kernel.shape[0], kernel_size=3, padding=1, bias=False).to(torch.cuda.current_device())
        self.sobel.weight.data.copy_(self.kernel)
        self.sobel.weight.requires_grad = False

    def forward(self, img):
        
        img_sobel = torch.zeros(list(img.shape)).repeat([1, self.kernel.shape[0], 1, 1]).to(torch.cuda.current_device())
        list(img.shape)
        for idx_chan in range(list(img.shape)[1]):
            img_sobel[:,idx_chan*2:idx_chan*2+2,:,:] += self.sobel(torch.unsqueeze(img[:,idx_chan,:,:], 1))
        return img_sobel
    
class Loss_Sobel3d(torch.nn.Module):
    def __init__(self,loss,channel_out):
        super(Loss_Sobel3d, self).__init__()
        self.loss = loss
        self.filt = Sobel3d(channel_out)
    
    def forward(self, output, target):
        #sobel3d = Sobel3d(channel_out)
        #output = sobel3d(output)
        #target = sobel3d(target)
        output = self.filt(output)
        target = self.filt(target)
        return self.loss(output,target)
    
class Loss_Sobel2d(torch.nn.Module):
    def __init__(self,loss):#,channel_out):
        super(Loss_Sobel2d, self).__init__()
        self.loss = loss
        self.filt = Sobel2d()#(channel_out)
    
    def forward(self, output, target):
        if len(target.shape) == 4:
            output = self.filt(output)
            target = self.filt(target)
            return self.loss(output,target)
        else:
            filtered_outputs = []
            filtered_targets = []
            # Get dimensions of input tensor
            batchs, channels, width, height, thickness = output.size()

            # Initialize empty list to store filtered tensors

            # Apply filt function to each 3D tensor along the fourth dimension
            for i in range(thickness):
                # Extract 3D tensor at the current index
                output_ = output[:, :, :, :, i]
                filtered_output = self.filt(output_)
                filtered_output = filtered_output.unsqueeze(4)
                target_ = target[:, :, :, :, i]
                filtered_target = self.filt(target_)
                filtered_target = filtered_target.unsqueeze(4)

                # Append filtered tensor to the list
                filtered_outputs.append(filtered_output)
                filtered_targets.append(filtered_target)

            # Concatenate filtered tensors along the fourth dimension
            output_f = torch.cat(filtered_outputs, dim=4)
            target_f = torch.cat(filtered_targets, dim=4)

            return self.loss(output_f,target_f)
    
class Loss_BCE(torch.nn.Module):
    def __init__(self):
        super(Loss_BCE, self).__init__()
        self.epsilon = 0.001
    def forward(self, output, target):
        output_rescale = (1-2*self.epsilon)*output+ self.epsilon
        #import pdb;pdb.set_trace()
        BCE = -1/target.shape[0]*torch.sum(target*torch.log(output_rescale)+(1-target)*torch.log((1-output_rescale)), dim = 0).squeeze()
        return BCE

if __name__ == "__main__":

    target = torch.randn(4, 1, 16, 244, 244)
    predicted = torch.randn(4, 1, 16, 244, 244).sigmoid()

    target[target > 0] = 1
    target[target < 0] = 0

    print(dice_coeff_loss(predicted, target))

    # target = torch.tensor([[[0,1,0],[1,1,1],[0,1,0]]], dtype=torch.float, requires_grad=True)
    # predicted = torch.tensor([[[1,1,0],[0,0,0],[1,0,0]]], dtype=torch.float, requires_grad=True)
    # print("Prediction : \n", predicted); print("GroudTruth : \n", target)
    # predicted.register_hook(get_grad)
    #
    # loss = TverskyLoss(0.3, torch.device("cpu"))
    # l = loss(predicted, target)
    # print("Loss : ", l)
    # l.backward()




def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):  ## works properly for isotropic 2D & if 3D it's summed over all z

    if len(img1.shape) == 5:
        img1 = torch.swapaxes(torch.swapaxes(img1,2,4),3,4)
        img1 = img1.contiguous().view(img1.shape[0],-1,*img1.shape[3:])#img1 = img1.flatten(1,2)
        img2 = torch.swapaxes(torch.swapaxes(img2,2,4),3,4)
        img2 = img2.contiguous().view(img2.shape[0],-1,*img2.shape[3:])#img2.flatten(1,2)
        

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class Loss_SSIM(torch.nn.Module): ## works properly for isotropic 2D & if 3D it's summed over all z
    def __init__(self, window_size=11, size_average=True, channel = 32, torch_device=None):
        super(Loss_SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        self.window = create_window(window_size, self.channel)
        self.torch_device = torch_device

    def forward(self, img1, img2):
        channel = img1.shape[1]#(_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.to(self.torch_device)
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return 1-_ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


class Loss_PCC(torch.nn.Module): ## works properly for isotropic 2D & if 3D it's summed over all z
    def __init__(self, eps = 1e-8, torch_device=None):
        super(Loss_PCC, self).__init__()
        self.torch_device = torch_device
        self.eps = eps

    def forward(self, img1, img2):
        tup_dim = tuple([i for i in range(2,len(img1.shape))])
        mu1 = torch.mean(img1, tup_dim)
        mu2 = torch.mean(img2, tup_dim)
        sigma1 = torch.std(img1, tup_dim)
        sigma2 = torch.std(img2, tup_dim)
        
        for i in range(2,len(img1.shape)):
            mu1 = mu1.unsqueeze(i)
            mu2 = mu2.unsqueeze(i)
            sigma1 = sigma1.unsqueeze(i)
            sigma2 = sigma2.unsqueeze(i)

        mu1 = mu1.repeat(1,1,*img1.shape[2:])
        mu2 = mu2.repeat(1,1,*img1.shape[2:])
        sigma1 = sigma1.repeat(1,1,*img1.shape[2:])
        sigma2 = sigma2.repeat(1,1,*img1.shape[2:])

        img1_ = (img1-mu1)/(sigma1+self.eps)
        img2_ = (img2-mu2)/(sigma2+self.eps)
        
        PCC = img1_*img2_
        return 1-PCC.mean()



if __name__ == '__main__':
    a = torch.randn(4, 128, 128, 128)
    b = torch.randn(4, 128, 128, 128)

    loss = ssim(a, b)

    print(loss)