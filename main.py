# -*- coding: UTF-8 -*-
import argparse
import copy
import os.path
import time
import numpy as np
import torch
import random
from torch.backends import cudnn
from torch.utils.data import DataLoader

from fed.client import create_clients
from fed.server import FedAvg
from utils.datasets import get_full_dataset
from utils.models import get_model
from utils.test import test_img
from utils.train import get_optim
from utils.utils import printf
import distutils.util

parser = argparse.ArgumentParser()
# global settings
parser.add_argument('--start_epochs', type=int, default=0, help='start epochs (only used in save model)')
parser.add_argument('--epochs', type=int, default=500, help="rounds of training")
parser.add_argument('--num_clients', type=int, default=50, help="number of clients: K")
parser.add_argument('--num_clients_each_iter', type=int, default=20, help="the fraction of clients: C")
parser.add_argument('--global_lr', type=float, default=1.0, help="global (or server's) learning rate")

# local settings
parser.add_argument('--local_ep', type=int, default=2, help="the number of local epochs: E")
parser.add_argument('--local_bs', type=int, default=128, help="local batch size: B")
parser.add_argument('--local_optim', type=str, default='sgd', help="local optimizer")
parser.add_argument('--local_lr', type=float, default=0.01, help="learning rate")
parser.add_argument('--local_momentum', type=float, default=0, help="SGD momentum (default: 0.5)")
parser.add_argument('--local_loss', type=str, default="CE", help="Loss Function")
parser.add_argument('--distribution', type=str, default='iid', help="the distribution used to split the dataset")

# test set settings
parser.add_argument('--test_bs', type=int, default=128, help="test batch size")

# model arguments
parser.add_argument('--model', type=str, default='VGG16', help='model name')
parser.add_argument('--num_classes', type=int, default=10, help="number of classes")

# other arguments
parser.add_argument('--dataset', type=str, default='cifar10', help="name of dataset")
parser.add_argument('--num_channels', type=int, default=3, help="number of channels of images")
parser.add_argument('--image_size', type=int, default=32, help="length or width of images")
parser.add_argument('--gpu', type=int, default=1, help="GPU ID, -1 for CPU")
parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
parser.add_argument('--verbose', action='store_true', help='verbose print')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--save_dir', type=str, default="./result/VGG16/50-20/")
parser.add_argument('--save', type=bool, default=True)


args = parser.parse_args()

if __name__ == '__main__':
    # log file path
    log_path = os.path.join(args.save_dir, 'log.txt')
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    # set seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        random.seed(args.seed)
        cudnn.deterministic = True
    # load dataset
    train_dataset, test_dataset = get_full_dataset(args.dataset, img_size=(args.image_size, args.image_size))
    # create clients
    clients = create_clients(args, train_dataset)
    # create global model
    global_model = get_model(args)

    # training
    train_loss = []
    val_loss, val_acc = [], []
    acc_best = None
    num_clients_each_iter = max(min(args.num_clients, args.num_clients_each_iter), 1)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    for epoch in range(args.epochs):
        start_time = time.time()
        local_losses = []
        local_models = []
        local_nums = []
        clients_idxs = np.random.choice(range(args.num_clients), num_clients_each_iter, replace=False)
        # train locally
        for idx in clients_idxs:
            current_client = clients[idx]
            local_model, num_samples, local_loss = current_client.train_one_iteration()
            local_models.append(copy.deepcopy(local_model))
            local_losses.append(local_loss)
            local_nums.append(num_samples)
        # aggregation
        global_model.load_state_dict(FedAvg(local_models, local_nums))

        # print loss
        avg_loss = sum(local_losses) / len(local_losses)
        printf('Round {:3d}, Average loss {:.3f}'.format(epoch, avg_loss), log_path)
        printf('Time: {}'.format(time.time() - start_time), log_path)
        train_loss.append(avg_loss)
        # send global model to clients
        for client in clients:
            client.set_model(copy.deepcopy(global_model))

        # testing
        global_model.eval()
        # acc_train, loss_train = test_img(global_model, train_dataset, args)
        # print("Training accuracy: {:.2f}".format(acc_train))
        acc_test, loss_test = test_img(global_model, test_dataset, args)
        printf("Testing accuracy: {:.2f}".format(acc_test), log_path)
        if acc_best is None or acc_best < acc_test:
            acc_best = acc_test
            if args.save:
                torch.save(global_model.state_dict(), args.save_dir + "model_best.pth")


    printf("Best Acc:" + str(acc_best), log_path)
    if args.save:
        torch.save(global_model.state_dict(),
                   args.save_dir + "model_last_epochs_" + str((args.epochs + args.start_epochs)) + ".pth")
