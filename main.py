import os 
import shutil
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from models import *
from utils import *

p = argparse.ArgumentParser()
p.add_argument('-b', '--batch_size', type=int, default=100,
               help='Training batch size.')
p.add_argument('-d', '--dataset', required=True, choices=['mnist', 'cifar10', 'svhn'])
p.add_argument('--checkpoints', type=str, default='checkpoints',
               help='Network models saving directory name.')
p.add_argument('--gpu', type=str, default='0',
               help='GPU number.')
p.add_argument('-e', '--epoch', type=int, default=100,
               help='Number of trainig epoch.')
p.add_argument('--lr', type=float, default=0.1)
p.add_argument('--weight_decay', type=float, default=5e-4)
p.add_argument('--momentum', type=float, default=0.9)
p.add_argument('--image_size', type=float, default=32)
p.add_argument('--model_type', type=str, default='cnn', choices=['cnn', 'resnet'])
p.add_argument('--seed_pytorch', type=int, default=np.random.randint(4294967295))
p.add_argument('--seed_numpy', type=int, default=np.random.randint(4294967295))
args = p.parse_args()
np.random.seed(args.seed_numpy)
torch.manual_seed(args.seed_pytorch)

os.makedirs(args.checkpoints, exist_ok=True)
tb_path = os.path.join(args.checkpoints, 'logs')
if os.path.exists(tb_path):
    shutil.rmtree(tb_path)
tb = SummaryWriter(tb_path)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device('cuda:0')
best_acc = 0

def main():
    global best_acc
    
    dataloader_train, dataloader_test, in_ch = get_dataset(args)
    if args.model_type == 'cnn':
        model = CNN(in_ch=in_ch, n_cls=10, img_size=args.image_size).to(device)
    elif args.model_type == 'resnet':
        model = ResNet(in_ch=in_ch, depth=18, classes=10).to(device)
    else:
        assert 0, 'Network type %s is not supported.' % args.network_type
    
    model = nn.DataParallel(model)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    cstm_lossfunc = CustomLossFunction()
    xent = nn.CrossEntropyLoss()
    iters = 0
    
    scheduler = [int(args.epoch*0.5), int(args.epoch*0.75)]
    adjust_learning_rate = lr_scheduler.MultiStepLR(optimizer, scheduler, gamma=0.1)
    
    for epoch in range(args.epoch):
        iters = train(epoch, model, dataloader_train, optimizer, cstm_lossfunc, iters)
        adjust_learning_rate.step()
        test_acc = validation(epoch, model, dataloader_test, xent)
        
        is_best = test_acc > best_acc
        if is_best:
            best_acc = max(test_acc, best_acc)
        
        save_checkpoint({
            'epoch': epoch,
            'model_type': args.model_type,
            'seed_numpy': args.seed_numpy,
            'seed_pytorch': args.seed_pytorch,
            'state_dict': model.state_dict(),
            'seed_numpy': args.seed_numpy,
            'seed_pytorch': args.seed_pytorch,
            'acc': test_acc,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict()}, is_best, checkpoint=args.checkpoints)
        
def train(epoch, model, dataloader, optimizer, cstm_lossfunc, iters):
    model.train()
    top1 = AverageMeter()
    losses = AverageMeter()
    for idx, (data, tgt) in enumerate(dataloader):
        data, tgt = data.to(device), tgt.to(device)
        b = data.size(0)
        gamma = np.random.beta(1,1)
                
        rand_idx = torch.randperm(b).to(device)
        data_rand = data[rand_idx]
        mixed_data = gamma * data + (1 - gamma) * data_rand
            
        onehot = torch.eye(10)[tgt].to(device)
        onehot_rand = onehot[rand_idx]
        mixed_tgt = gamma * onehot + (1 - gamma) * onehot_rand
            
        optimizer.zero_grad()
        logits = model(mixed_data)
        loss = cstm_lossfunc.xent(logits, mixed_tgt)
        loss.backward()
        optimizer.step()
            
        iters += 1
        if idx % 100 == 0:
            prec1 = accuracy(logits.data, tgt, topk=(1,))[0]
            losses.update(loss.data.item(), b)
            top1.update(prec1, b)
            print('%d epochs [%d/%d]| loss: %.4f | acc: %.4f |' % (epoch, idx, len(dataloader), loss.item(), top1.avg))
            tb.add_scalar('train loss', losses.avg, iters)
            tb.add_scalar('train acc', top1.avg, iters)
    return iters
                      
def validation(epoch, model, dataloader, xent):
    model.eval()
    top1 = AverageMeter()
    losses = AverageMeter()
    for idx, (data, tgt) in enumerate(dataloader):
        data, tgt = data.to(device), tgt.to(device)
            
        with torch.no_grad():
            logits = model(data)
                
        loss = xent(logits, tgt)    
        prec1 = accuracy(logits.data, tgt, topk=(1,))[0]
        losses.update(loss.data.item(), data.size(0))
        top1.update(prec1, data.size(0))
    print('%d epochs | loss: %.4f | acc: %.4f |' % (epoch, loss.item(), top1.avg))
    tb.add_scalar('test loss', losses.avg, epoch)
    tb.add_scalar('test acc', top1.avg, epoch)
    return top1.avg
        
        
if __name__ == '__main__':
    main()
            
            
            
            
            
        