import os
import torch
import shutil
import multiprocessing
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

def get_dataset(config):
    transform_mnist = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor()])
    transform_other = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
    
    if config.dataset == 'mnist':
        train_data = datasets.__dict__[config.dataset.upper()](root='./data', train=True, transform=transform_mnist, download=True)
        test_data = datasets.__dict__[config.dataset.upper()](root='./data', train=True, transform=transform_mnist, download=True)
        in_ch = 1
    elif config.dataset == 'cifar10':
        train_data = datasets.__dict__[config.dataset.upper()](root='./data', train=True, transform=transform_other, download=True)
        test_data = datasets.__dict__[config.dataset.upper()](root='./data', train=True, transform=transform_other, download=True)
        in_ch = 3
    elif config.dataset == 'svhn':
        training_data = datasets.__dict__[config.dataset.upper()](root='./data', split='train', transform=transform_other, download=True)
        test_data = datasets.__dict__[config.dataset.upper()](root='./data', split='test', transform=transform_other, download=True)
        in_ch = 3
    else:
        assert 0, 'Dataset %s is not supported.' % conf.training_data_name
    dataloader_train = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, num_workers=multiprocessing.cpu_count())
    dataloader_test = DataLoader(test_data, batch_size=config.batch_size, shuffle=False, num_workers=multiprocessing.cpu_count())
    return dataloader_train, dataloader_test, in_ch

class CustomLossFunction:
    def __init__(self, reduction='mean'):
        self.reduction = reduction
        
    def xent(self, x, t):
        b, c = x.shape
        x_log_softmax = torch.log_softmax(x, dim=1)
        if self.reduction == 'mean':
            loss = -torch.sum(t*x_log_softmax) / b
        elif self.reduction == 'sum':
            loss = -torch.sum(t*x_log_softmax)
        elif self.reduction == 'none':
            loss = -torch.sum(t*x_log_softmax, keepdims=True)
        return loss
    
def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    print('Model save..')
    torch.save(state, filepath)
    if is_best:
        print('==> Updating the best model..')
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))
    
##########################################################################
# the function to calclate accuracy
##########################################################################
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.contiguous().view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res