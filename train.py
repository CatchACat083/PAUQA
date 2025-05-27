import argparse
import random
from argparse import ArgumentParser

import os
import numpy as np
import random

import logging
from tqdm import tqdm

import torch
from torch import nn
from torch import optim

from utiles import *
from dataloader import *
import datetime

from models.uiqa.pauqa_model import PAUQA 


try:
    from tensorboardX import SummaryWriter
except ImportError:
    raise RuntimeError("No tensorboardX package is found. Please install with the command: \npip install tensorboardX")

current_time = datetime.datetime.now().strftime("%m-%d-%Y-%H-%M-%S")


def train_model(args, model, train_loader, val_loader, writer,  
                device, loss_type,
                epochs:int=5, batch_size:int=1, learning_rate:float=1e-5,
                amp:bool=False, weight_decay:float=1e-5, momentum:float=0.9, 
                gradient_clipping: float = 1.0,
                schduler_decay_interval:int=50, schduler_decay_ratio:float=0.8,
                save_iter_checkpoint:bool=False,  save_best_checkpoint:bool=False,  save_final_checkpoint:bool=False
):

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Weight decay:    {weight_decay}
        Best Checkpoints:     {save_best_checkpoint}
        Liter Checkpoints:    {save_iter_checkpoint}
        Device:          {device.type}
        Mixed Precision: {amp}
    ''')

    # >>> 1. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    if args.optim_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(momentum, 0.999), foreach=True)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(momentum, 0.999), foreach=True)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=schduler_decay_interval, gamma=schduler_decay_ratio)     # after every "decay_interval", lr = lr * decay_ratio
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)

    # >>> 2. Set up loss function.
    if args.loss_type == 'l1':
        loss_function = nn.L1Loss()
    else:
        loss_function = nn.L1Loss()


    # >>> 3. Set up best validation criterion.
    global best_val_criterion, best_epoch
    if args.val_criterion == "RMSE":
        best_val_criterion, best_epoch = 10000, -1  # larger, better, e.g., SROCC or PLCC. If RMSE is used, best_val_criterion <- 10000
    else:
        best_val_criterion, best_epoch = -1, -1  # larger, better, e.g., SROCC or PLCC. If RMSE is used, best_val_criterion <- 10000
    best_criterion = {}


    # >>> 4. Set up global step.
    global_step = 0

    # >>> 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch}/{epochs}', unit='batch', ncols=150) as pbar:
            for batch in train_loader:
                # >>> ITERATION STEP.
                images, scores = batch
                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                scores = scores.to(device=device, dtype=torch.float32)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    # >>> Predict the score.
                    outputs = model(images)

                    # >>>  Calculate loss.
                    loss = loss_function(outputs, scores.reshape((-1, 1)))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update(1)
                global_step += 1
                epoch_loss += loss.item()

                # >>> ITERATION_COMPLETED.
                writer.add_scalar("training/iteration_loss", loss.item(), global_step)
                pbar.set_postfix(**{'loss (batch)': loss.item(), 'lr': optimizer.param_groups[0]['lr']})

        # >>> EPOCH_COMPLETED.
        performance = evaluate_model(model, val_loader, device, amp)
        val_srocc, val_krocc, val_plcc, val_rmse, val_mae = \
                    performance['SROCC'], performance['KROCC'], performance['PLCC'], \
                    performance['RMSE'], performance['MAE']                     
        val_criterion = abs(performance[args.val_criterion]) 

        scheduler.step()

        logging.info('Validation SROCC: {:.4f}, KROCC: {:.4f}, PLCC: {:.4f}, RMSE: {:.4f}, MAE: {:.4f}'.format(val_srocc, val_krocc, val_plcc, val_rmse, val_mae))
        
        writer.add_scalar("training/epoch_loss", epoch_loss, epoch)
        writer.add_scalar("validation/srocc", val_srocc, epoch)
        writer.add_scalar("validation/krocc", val_krocc, epoch)
        writer.add_scalar("validation/plcc", val_plcc, epoch)
        writer.add_scalar("validation/rmse", val_rmse, epoch)
        writer.add_scalar("validation/mae", val_mae, epoch)
        writer.add_scalar("training/learning_rate", optimizer.param_groups[0]['lr'], epoch)


        # >>> save iteration model.
        if save_iter_checkpoint:
            # save checkpoint every 10 epoch.
            if epoch % 20 == 0:
                checkpoint = {
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'epoch': epoch
                            }
                 #torch.save(checkpoint, args.checkpoint_model_file)
                os.makedirs('results/{}'.format(current_time), exist_ok=True)
                checkpoint_model_file = 'results/{}/{}-{}-lr={}-ft_lr_ratio={}-bs={}-epoch={}.pth'.format(current_time, args.model, args.database, args.learning_rate, args.ft_lr_ratio, args.batch_size, epoch)
                torch.save(checkpoint, checkpoint_model_file)
                logging.info('>>> Save model criterion: @epoch: {}'.format(epoch))


        # >>> save best model.
        if args.val_criterion == "RMSE":
            if val_criterion < best_val_criterion: # If RMSE is used, then change ">" to "<".
                best_criterion = {  'epoch':        epoch,      'srocc':        val_srocc,
                                    'krocc':        val_krocc,  'plcc':         val_plcc,
                                    'rmse':         val_rmse,   'mae':          val_mae,
                                    }
                best_val_criterion = val_criterion
                best_epoch = epoch
                print('>>> Save current best model @best_val_criterion ({}): {:.5f} @epoch: {}'.format(args.val_criterion, best_val_criterion, best_epoch))

                if args.save_best == True:
                    checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }
                    torch.save(checkpoint, args.trained_model_file)
            else:
                logging.info('Model is not updated @val_criterion ({}): {:.5f} @epoch: {}'.format(args.val_criterion, val_criterion, epoch))
        
        else:
            if val_criterion > best_val_criterion: # If RMSE is used, then change ">" to "<".
                best_criterion = {  'epoch':        epoch,      'srocc':        val_srocc,
                                    'krocc':        val_krocc,  'plcc':         val_plcc,
                                    'rmse':         val_rmse,   'mae':          val_mae,
                                    }
                if args.save_best == True:
                    checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch
                    }
                    torch.save(checkpoint, args.trained_model_file)
                best_val_criterion = val_criterion
                best_epoch = epoch
                logging.info('>>> Save current best model @best_val_criterion ({}): {:.5f} @epoch: {}'.format(args.val_criterion, best_val_criterion, best_epoch))
            else:
                logging.info('Model is not updated @val_criterion ({}): {:.5f} @epoch: {}'.format(args.val_criterion, val_criterion, epoch))

    # TRAINING_COMPLETED.
    performance = evaluate_model(model, val_loader, device, amp)
    val_srocc, val_krocc, val_plcc, val_rmse, val_mae = \
                performance['SROCC'], performance['KROCC'], performance['PLCC'], \
                performance['RMSE'], performance['MAE']             

    val_criterion = abs(performance[args.val_criterion]) 
    logging.info('*Final* Validation SROCC: {:.4f}, KROCC: {:.4f}, PLCC: {:.4f}, RMSE: {:.4f}, MAE: {:.4f}'.format(val_srocc, val_krocc, val_plcc, val_rmse, val_mae))
    logging.info('*Best* Validataion SROCC: {:.4f}, KROCC: {:.4f}, PLCC: {:.4f}, RMSE: {:.4f}, MAE: {:.4f} at epoch: {}'.format(best_criterion['srocc'], best_criterion['krocc'], \
                                                                                                                                best_criterion['plcc'], best_criterion['rmse'], \
                                                                                                                                best_criterion['mae'], best_criterion['epoch']))
    logging.info('at dict:{}'.format(current_time))


    if save_final_checkpoint:
        checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch
                        }
        torch.save(checkpoint, args.final_model_file)
        logging.info('Save final model criterion: {:.5f} @epoch: {}'.format(val_criterion, epoch))
        
    writer.add_text("validation/best", '*Best* Validataion SROCC: {:.4f}, KROCC: {:.4f}, PLCC: {:.4f}, RMSE: {:.4f}, MAE: {:.4f} at epoch: {}'.format(best_criterion['srocc'], best_criterion['krocc'], \
                                                                                                                                best_criterion['plcc'], best_criterion['rmse'], \
                                                                                                                                best_criterion['mae'], best_criterion['epoch']))
    
    writer.close()


def evaluate_model(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)

    metric = UWIQAPerformance(status='train')
    metric.reset()

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        with torch.no_grad():
            for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=True, ncols=150):
                images, scores = batch

                # move images and labels to correct device and type
                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                scores = scores.to(device=device, dtype=torch.float32)

                outputs = net(images)
                metric.update((outputs, scores))

    performance = metric.compute()

    net.train()

    return performance


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch Underwater IQA')
    parser.add_argument("--seed", type=int, default=1234567)

    parser.add_argument('--database', default='LUIQD', type=str, help='database name')
    parser.add_argument('--model', default='PAUQA', type=str, help='model name')

    parser.add_argument('--pretrained', default=None, type=str, help='path to latest checkpoint (default: None)')

    parser.add_argument('--val_criterion', default='SROCC', type=str, help='val_criterion: RMSE or SROCC or PLCC') # If using RMSE, minor modification should be made, i.e., 
    parser.add_argument('--loss_type', default='l1', type=str, help='loss type')
    parser.add_argument('--optim_type', default='adam', type=str, help='optimizer type')

    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--ft_lr_ratio', type=float, default=0.1, help='ft_lr_ratio')
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--decay_interval', type=int, default=100, help='learning rate decay interval')
    parser.add_argument('--decay_ratio', type=float, default=0.5, help='learning rate decay ratio')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')


    parser.add_argument("--log_dir", type=str, default="tensorboard_logs", help="log directory for Tensorboard log output")
    parser.add_argument('--save_best', default=True, type=bool, help='Save the best model checkpoints during training.')
    parser.add_argument('--save_iter', default=False, type=bool, help='Save the iteration model checkpoints during training.')
    parser.add_argument('--save_final', default=False, type=bool, help='Save the final model checkpoints during training.')

    parser.add_argument('--disable_gpu', action='store_true', help='flag whether to disable GPU')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    device = torch.device("cuda" if not args.disable_gpu and torch.cuda.is_available() else "cpu")

    # >>> [checkpoints] set checkpoints and results dir.
    # >>> set checkpoints and results dir.
    args.log_format_str = '{}/{}-{}-loss={}-lr={}-ft_lr_ratio={}-bs={}-{}'.format(args.log_dir, args.model, args.database, args.loss_type, args.learning_rate, args.ft_lr_ratio, args.batch_size, current_time)
    if args.save_best:
        os.makedirs('checkpoints/{}/{}'.format(args.model, current_time,), exist_ok=True)
        args.trained_model_file = 'checkpoints/{}/{}-{}-loss={}-lr={}-ft_lr_ratio={}-bs={}-{}-best'.format(args.model, args.model, args.database, args.loss_type, args.learning_rate, args.ft_lr_ratio, args.batch_size, current_time)
    if args.save_final:
        os.makedirs('results/{}/{}'.format(args.model, current_time,), exist_ok=True)
        args.final_model_file = 'results/{}/{}-{}-loss={}-lr={}-ft_lr_ratio={}-bs={}-{}-final'.format(args.model, args.model, args.database, args.loss_type, args.learning_rate, args.ft_lr_ratio, args.batch_size, current_time)
        args.save_result_file = 'results/{}/{}-{}-loss={}-lr={}-ft_lr_ratio={}-bs={}-{}'.format(args.model, args.model, args.database, args.loss_type, args.learning_rate, args.ft_lr_ratio, args.batch_size, current_time)

    # >>> [random] set random seeds info.
    # torch.manual_seed(args.seed)  
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # np.random.seed(args.seed)
    # random.seed(args.seed)

    # >>> [logs] set logging info.
    args.tensorboard_dir = 'tensorboardLogs' 
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info(f'Using device {device}')

    writer = SummaryWriter(log_dir='{}'.format(args.log_format_str))

    # >>> [model] set model info.
    # >>>> <1> set models. >>>>> 
    if args.model == 'PAUQA':
        model = PAUQA(img_size=384, num_classes=1, num_stages=4,  
                        num_paths=[2, 3, 3, 3], patch_size=[[3,3],[3,3,3],[3,3,3],[3,3,3]], dilation_size=[[1,2],[1,2,3],[1,2,3],[1,2,3]],
                        blk_depths=[1, 2, 4, 1], embed_dims=[64, 128, 192, 256], # res_dims=[3, 4, 6, 3],
                        mlp_ratios=[4, 4, 4, 4], num_heads=[8, 8, 8, 8])
        
    model = model.to(memory_format=torch.channels_last)

    # >>> [dataloader] set train data loader. >>>>>
    train_loader, val_loader = data_loader(dataset_name = args.database, batch_size = args.batch_size)

    # >>> [pretrained] reload pretrained models
    if args.pretrained:
        ckpt = torch.load(args.pretrained)
        model.load_state_dict(ckpt["model"])

    model.to(device=device)

    train_model(args=args, model=model, train_loader=train_loader, val_loader=val_loader, writer=writer,
                device=device, loss_type=args.loss_type,
                epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
                amp=args.amp, weight_decay=args.ft_lr_ratio,
                schduler_decay_interval=args.decay_interval, schduler_decay_ratio=args.decay_ratio,
                save_best_checkpoint=args.save_best, save_iter_checkpoint=args.save_iter, save_final_checkpoint=args.save_final,
                )
    

