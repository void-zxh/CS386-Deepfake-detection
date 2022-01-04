import sys
import os
import argparse
import random
import time
from typing import final
import warnings

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
warnings.filterwarnings('ignore')

#from apex import amp
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim
from thop import profile
from thop import clever_format
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from config import Config
import os
import numpy as np
import torch
import models
import logging
from logging.handlers import TimedRotatingFileHandler


def get_logger(name, log_dir='log'):
    """
    Args:
        name(str): name of logger
        log_dir(str): path of log
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    info_name = os.path.join(log_dir, '{}.info.log'.format(name))
    info_handler = TimedRotatingFileHandler(info_name,
                                            when='D',
                                            encoding='utf-8')
    info_handler.setLevel(logging.INFO)
    error_name = os.path.join(log_dir, '{}.error.log'.format(name))
    error_handler = TimedRotatingFileHandler(error_name,
                                             when='D',
                                             encoding='utf-8')
    error_handler.setLevel(logging.ERROR)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    info_handler.setFormatter(formatter)
    error_handler.setFormatter(formatter)

    logger.addHandler(info_handler)
    logger.addHandler(error_handler)

    return logger


class DataPrefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            sample = next(self.loader)
            self.next_input, self.next_target, self.next_mask = sample
        except StopIteration:
            self.next_input = None
            self.next_target = None
            self.next_mask = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            self.next_mask = self.next_mask.cuda(non_blocking=True)
            self.next_input = self.next_input.float()
            self.next_mask = self.next_mask.float()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        mask = self.next_mask
        self.preload()
        return input, target, mask


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


def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

        return res


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--network',
                        type=str,
                        default=Config.network,
                        help='name of network')
    parser.add_argument('--loss_lambda',
                        type=int,
                        default=Config.loss_lambda,
                        help='Loss lambda')
    parser.add_argument('--lr',
                        type=float,
                        default=Config.lr,
                        help='learning rate')
    parser.add_argument('--momentum',
                        type=float,
                        default=Config.momentum,
                        help='momentum')
    parser.add_argument('--weight_decay',
                        type=float,
                        default=Config.weight_decay,
                        help='weight decay')
    parser.add_argument('--epochs',
                        type=int,
                        default=Config.epochs,
                        help='num of training epochs')
    parser.add_argument('--batch_size',
                        type=int,
                        default=Config.batch_size,
                        help='batch size')
    parser.add_argument('--milestones',
                        type=list,
                        default=Config.milestones,
                        help='optimizer milestones')
    parser.add_argument('--accumulation_steps',
                        type=int,
                        default=Config.accumulation_steps,
                        help='gradient accumulation steps')
    parser.add_argument('--pretrained',
                        type=bool,
                        default=Config.pretrained,
                        help='load pretrained model params or not')
    parser.add_argument('--num_classes',
                        type=int,
                        default=Config.num_classes,
                        help='model classification num')
    parser.add_argument('--input_image_size',
                        type=int,
                        default=Config.input_image_size,
                        help='input image size')
    parser.add_argument('--num_workers',
                        type=int,
                        default=Config.num_workers,
                        help='number of worker to load data')
    parser.add_argument('--resume',
                        type=str,
                        default=Config.resume,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkpoints',
                        type=str,
                        default=Config.checkpoint_path,
                        help='path for saving trained models')
    parser.add_argument('--log',
                        type=str,
                        default=Config.log,
                        help='path to save log')
    parser.add_argument('--evaluate',
                        type=str,
                        default=Config.evaluate,
                        help='path for evaluate model')
    parser.add_argument('--test',
                        type=str,
                        default=Config.test,
                        help='path for test model')
    parser.add_argument('--video_test',
                        type=str,
                        default=Config.video_test,
                        help='path for video_test model')
    parser.add_argument('--seed', type=int, default=Config.seed, help='seed')
    parser.add_argument('--print_interval',
                        type=bool,
                        default=Config.print_interval,
                        help='print interval')
    parser.add_argument('--apex',
                        type=bool,
                        default=Config.apex,
                        help='use apex or not')

    return parser.parse_args()


def train(train_loader, model, criterion, optimizer, scheduler, epoch, logger,
          args):
    top1 = AverageMeter()
    top5 = AverageMeter()
    auc=AverageMeter()
    losses = AverageMeter()
    criterionBCE=nn.BCELoss().cuda()
    Softm=nn.Softmax(dim=1)
    # switch to train mode
    model.train()

    iters = len(train_loader.dataset) // args.batch_size
    loss_lambda=args.loss_lambda
    prefetcher = DataPrefetcher(train_loader)
    inputs, labels, masks = prefetcher.next()
    iter_index = 1
    while inputs is not None:
        inputs, labels, masks = inputs.cuda(), labels.cuda(), masks.cuda()
        #print(inputs.shape)
        outputs,output_mask = model(inputs)
        sz=output_mask.shape
        #print(sz)

        masks_p=masks.clone()
        masks=masks.repeat(1,sz[1]*sz[2],1,1)
        masks=masks.view(sz)
        # print(masks[0][0][0])
        # print(masks[0][0][1])
        masks_p=torch.flatten(masks_p,2)
        masks_p=torch.transpose(masks_p,1,2)
        masks_p=masks_p.repeat(1,1,sz[1]*sz[2])
        m_sz=masks_p.shape
        masks_p=masks_p.view(m_sz[0],m_sz[1],sz[1],sz[2])
        masks_p=masks_p.view(m_sz[0],sz[1],sz[2],sz[1],sz[2])
        masks=1-torch.abs(masks_p-masks)

        #print(masks_p.shape)
        #print(sz)
        #print(outputs)
        #print(labels)
        CLS_loss = criterion(outputs, labels)
        CLS_loss = CLS_loss / args.accumulation_steps
        PCL_loss=criterionBCE(output_mask,masks)
        #print(CLS_loss)
        #print(PCL_loss)
        loss=CLS_loss+loss_lambda*PCL_loss
        #print(loss)
        if args.apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        if iter_index % args.accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        # measure accuracy and record loss
        acc1, acc5 = accuracy(outputs, labels, topk=(1, 2))
        top1.update(acc1.item(), inputs.size(0))
        top5.update(acc5.item(), inputs.size(0))
        y_true=labels.cpu().detach().numpy()
        count_label=np.unique(y_true)
        #print(y_true.shape)
        output_score=Softm(outputs)
        #output_score=outputs
        #print(outputs)
        y_score=output_score.cpu().detach().numpy()
        #print(y_score)
        #print(y_score.shape)
        y_score=y_score[:,1]
        #print(y_score.shape)
        no_auc=(len(count_label)==1)
        if no_auc ==0:
            re_auc=roc_auc_score(y_true, y_score)
            auc.update(re_auc,inputs.size(0))
        losses.update(loss.item(), inputs.size(0))

        inputs, labels,masks = prefetcher.next()

        if iter_index % args.print_interval == 0:
            if no_auc==0:
                logger.info(
                    f"train: epoch {epoch:0>3d}, iter [{iter_index:0>4d}, {iters:0>4d}], lr: {scheduler.get_lr()[0]:.6f}, this batch auc: {re_auc.item():.5f}, top1 acc: {acc1.item():.2f}%, top5 acc: {acc5.item():.2f}%, loss_total: {loss.item():.2f}"
                )
            else:
                logger.info(
                    f"train: epoch {epoch:0>3d}, iter [{iter_index:0>4d}, {iters:0>4d}], lr: {scheduler.get_lr()[0]:.6f}, this batch auc: nope, top1 acc: {acc1.item():.2f}%, top5 acc: {acc5.item():.2f}%, loss_total: {loss.item():.2f}"
                )

        iter_index += 1

    scheduler.step()

    return top1.avg, top5.avg,auc.avg, losses.avg


def videotest(val_loader, model, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    Softm=nn.Softmax(dim=1)
    # switch to evaluate mode
    model.eval()
    final_y_true=None
    final_y_score=None
    first_in=0

    with torch.no_grad():
        end = time.time()
        for inputs, labels, video_names in val_loader:
            data_time.update(time.time() - end)
            inputs, labels, video_names= inputs.cuda(), labels.cuda(), video_names.cuda()
            outputs,output_mask = model(inputs)
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 2))
            top1.update(acc1.item(), inputs.size(0))
            top5.update(acc5.item(), inputs.size(0))
            y_true=labels.cpu().detach().numpy()
            #print(y_true.shape)
            #output_score=outputs
            #print(output_score)
            output_score=Softm(outputs)
            y_score=output_score.cpu().detach().numpy()
            y_score=y_score[:,1]
            if first_in==0:
                final_y_true=np.copy(y_true)
            else:
                final_y_true=np.concatenate((final_y_true,y_true),axis=0)
            if first_in==0:
                final_y_score=np.copy(y_score)
                first_in=1
            else:
                final_y_score=np.concatenate((final_y_score,y_score),axis=0)
            batch_time.update(time.time() - end)
            end = time.time()

    throughput = 1.0 / (batch_time.avg / inputs.size(0))
    re_auc=roc_auc_score(final_y_true, final_y_score)

    return top1.avg, top5.avg,re_auc, throughput

def validate(val_loader, model, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    Softm=nn.Softmax(dim=1)
    # switch to evaluate mode
    model.eval()
    final_y_true=None
    final_y_score=None
    first_in=0

    with torch.no_grad():
        end = time.time()
        for inputs, labels in val_loader:
            data_time.update(time.time() - end)
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs,output_mask = model(inputs)
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 2))
            top1.update(acc1.item(), inputs.size(0))
            top5.update(acc5.item(), inputs.size(0))
            y_true=labels.cpu().detach().numpy()
            #print(y_true.shape)
            #output_score=outputs
            #print(output_score)
            output_score=Softm(outputs)
            y_score=output_score.cpu().detach().numpy()
            y_score=y_score[:,1]
            print(y_score.shape)
            if first_in==0:
                final_y_true=np.copy(y_true)
            else:
                final_y_true=np.concatenate((final_y_true,y_true),axis=0)
            if first_in==0:
                final_y_score=np.copy(y_score)
                first_in=1
            else:
                final_y_score=np.concatenate((final_y_score,y_score),axis=0)
            batch_time.update(time.time() - end)
            end = time.time()

    throughput = 1.0 / (batch_time.avg / inputs.size(0))
    re_auc=roc_auc_score(final_y_true, final_y_score)

    return top1.avg, top5.avg,re_auc, throughput


def main(logger, args):
    if not torch.cuda.is_available():
        raise Exception("need gpu to train network!")

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True

    gpus = torch.cuda.device_count()
    logger.info(f'use {gpus} gpus')
    logger.info(f"args: {args}")

    cudnn.benchmark = True
    cudnn.enabled = True
    start_time = time.time()

    # dataset and dataloader
    logger.info('start loading data')
    print(Config.train_dataset.imgs)
    train_loader = DataLoader(Config.train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=args.num_workers)
    val_loader = DataLoader(Config.val_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=args.num_workers)
    test_loader = DataLoader(Config.test_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=args.num_workers)
    video_test_loader = DataLoader(Config.video_test_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=args.num_workers)
    logger.info('finish loading data')

    logger.info(f"creating model '{args.network}'")
    model = models.__dict__[args.network](**{
        "pretrained": args.pretrained,
        "num_classes": args.num_classes,
    })

    flops_input = torch.randn(1, 3, args.input_image_size,
                              args.input_image_size)
    flops, params = profile(model, inputs=(flops_input, ))
    flops, params = clever_format([flops, params], "%.3f")
    logger.info(f"model: '{args.network}', flops: {flops}, params: {params}")

    for name, param in model.named_parameters():
        logger.info(f"{name},{param.requires_grad}")

    model = model.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    # optimizer = torch.optim.SGD(model.parameters(),
    #                             args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr,betas=(0.9,0.999),eps=1e-8)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.milestones, gamma=0.1)

    if args.apex:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    model = nn.DataParallel(model)

    if args.evaluate:
        if not os.path.isfile(args.evaluate):
            raise Exception(
                f"{args.resume} is not a file, please check it again")
        logger.info('start only evaluating')
        logger.info(f"start resuming model from {args.evaluate}")
        checkpoint = torch.load(args.evaluate,
                                map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        acc1, acc5, auc, throughput = validate(val_loader, model, args)
        logger.info(
            f"epoch {checkpoint['epoch']:0>3d}, auc: {auc:.5f}, top1 acc: {acc1:.2f}%, top5 acc: {acc5:.2f}%, throughput: {throughput:.2f}sample/s"
        )

        return
    if args.test:
        if not os.path.isfile(args.test):
            raise Exception(
                f"{args.resume} is not a file, please check it again")
        logger.info('start only evaluating')
        logger.info(f"start resuming model from {args.test}")
        checkpoint = torch.load(args.test,
                                map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        #model.module.load_state_dict(checkpoint)
        acc1, acc5, auc, throughput = validate(test_loader, model, args)
        logger.info(
            f" auc: {auc:.5f}, top1 acc: {acc1:.2f}%, top5 acc: {acc5:.2f}%, throughput: {throughput:.2f}sample/s"
        )

        return

    if args.video_test:
        if not os.path.isfile(args.video_test):
            raise Exception(
                f"{args.resume} is not a file, please check it again")
        logger.info('start only evaluating')
        logger.info(f"start resuming model from {args.video_test}")
        checkpoint = torch.load(args.video_test,
                                map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        #model.module.load_state_dict(checkpoint)
        acc1, acc5, auc, throughput = videotest(video_test_loader, model, args)
        logger.info(
            f" auc: {auc:.5f}, top1 acc: {acc1:.2f}%, top5 acc: {acc5:.2f}%, throughput: {throughput:.2f}sample/s"
        )

        return

    start_epoch = 1
    # resume training
    if os.path.exists(args.resume):
        logger.info(f"start resuming model from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
        start_epoch += checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logger.info(
            f"finish resuming model from {args.resume}, epoch {checkpoint['epoch']}, "
            f"loss: {checkpoint['loss']:3f}, lr: {checkpoint['lr']:.6f}, "
            f"top1_acc: {checkpoint['acc1']}%")

    if not os.path.exists(args.checkpoints):
        os.makedirs(args.checkpoints)

    logger.info('start training')
    for epoch in range(start_epoch, args.epochs + 1):
        acc1, acc5,auc, losses = train(train_loader, model, criterion, optimizer,
                                   scheduler, epoch, logger, args)
        logger.info(
            f"train: epoch {epoch:0>3d}, top1 acc: {acc1:.2f}%, top5 acc: {acc5:.2f}%， losses: {losses:.2f}"
        )

        acc1, acc5,auc, throughput = validate(val_loader, model, args)
        logger.info(
            f"val: epoch {epoch:0>3d}, auc: {auc:.5f}, top1 acc: {acc1:.2f}%, top5 acc: {acc5:.2f}%, throughput: {throughput:.2f}sample/s"
        )

        # remember best prec@1 and save checkpoint
        torch.save(
            {
                'epoch': epoch,
                'acc1': acc1,
                'loss': losses,
                'lr': scheduler.get_lr()[0],
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, os.path.join(args.checkpoints, 'latest.pth'))
        if epoch == args.epochs:
            torch.save(
                model.module.state_dict(),
                os.path.join(
                    args.checkpoints,
                    "{}-epoch{}-acc{}.pth".format(args.network, epoch, acc1)))

    training_time = (time.time() - start_time) / 3600
    logger.info(
        f"finish training, total training time: {training_time:.2f} hours")


if __name__ == '__main__':
    args = parse_args()
    print(1)
    logger = get_logger(__name__, args.log)
    main(logger, args)
