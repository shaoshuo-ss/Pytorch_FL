# -*- coding: UTF-8 -*-
import argparse
import distutils.util
import os.path

from attack.finetune import *
from utils.datasets import get_full_dataset
from utils.test import test_img
from watermark.fingerprint import *
import torch
from utils.models import get_model

parser = argparse.ArgumentParser()

parser.add_argument('--start_epochs', type=int, default=0, help='start epochs (only used in save model)')
parser.add_argument('--epochs', type=int, default=500, help="rounds of training")
parser.add_argument('--num_clients', type=int, default=50, help="number of clients: K")
parser.add_argument('--num_clients_each_iter', type=int, default=2, help="the fraction of clients: C")
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
parser.add_argument('--gpu', type=int, default=-1, help="GPU ID, -1 for CPU")
parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
parser.add_argument('--verbose', action='store_true', help='verbose print')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--save_dir', type=str, default="./result/VGG16/5/")
parser.add_argument('--save', type=bool, default=True)

# watermark arguments
parser.add_argument("--watermark", type=lambda x: bool(distutils.util.strtobool(x)), default=True,
                    help="whether embedding the watermark and fingerprints")
parser.add_argument('--gfp_length', type=int, default=512, help="Bit length of global fingerprint")
parser.add_argument('--lfp_length', type=int, default=512, help="Bit length of local fingerprints")
parser.add_argument('--fp_threshold', type=float, default=0.5)
parser.add_argument('--num_trigger_set', type=int, default=100, help='number of images used as trigger set')
parser.add_argument('--embed_layer', type=str, default='model.bn8')
parser.add_argument('--lambda1', type=float, default=0.1)
parser.add_argument('--max_iters', type=int, default=10)
parser.add_argument('--lambda2', type=float, default=0.005)

args = parser.parse_args()


def fingerprint_fine_tune_attack(model, layer_name):
    model = fingerprint_fine_tune(model, layer_name, length=512, lr=0.005, epochs=5)
    embed_layer_name = layer_name.split('.')
    layer = model
    for name in embed_layer_name:
        layer = getattr(layer, name)
    return model


if __name__ == '__main__':
    model = get_model(args)
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    # layer_name = 'extractor.norm2'
    num_trace = 0
    total_ber = 0
    weight_size = 512
    base_path = "./result/VGG16/50/"
    global_fingerprint = generate_fingerprint(args.lfp_length)
    extracting_matrices = generate_extracting_matrices(weight_size, args.lfp_length, args.num_clients)
    train_dataset, test_dataset = get_full_dataset(args.dataset, img_size=(args.image_size, args.image_size))
    for client_idx in range(args.num_clients):
        path = os.path.join(base_path, "model_{}.pth".format(client_idx))
        model.load_state_dict(torch.load(path))
        # layer = model
        # for name in args.embed_layer.split('.'):
        #     layer = getattr(layer, name)
        # ber, extract_idx = extracting_fingerprints(layer, local_fingerprints, extracting_matrices)
        # print("Result_idx:{}, HD:{}".format(extract_idx, ber))
        # ber, extract_idx = extracting_fingerprints(layer, local_fingerprints, extracting_matrices)
        # print("Result_idx:{}, BER:{}".format(extract_idx, ber))
        fingerprint_fine_tune_attack(model, args.embed_layer)
        layer = model
        for name in args.embed_layer.split('.'):
            layer = getattr(layer, name)
        ber, extract_idx = extracting_global_fingerprints(layer, global_fingerprint, extracting_matrices)
        print("Result_idx:{}, HD:{}".format(extract_idx, ber))
        if extract_idx == client_idx:
            num_trace += 1
        total_ber += ber
        # acc_test, loss_test = test_img(model, test_dataset, args)
        # print("Testing accuracy: {:.2f}".format(acc_test))
    trace_rate = num_trace / args.num_clients
    avg_ber = total_ber / args.num_clients
    print("Trace rate:{}, average ber:{}".format(trace_rate, avg_ber))
    # global_fingerprint = generate_fingerprint(args.lfp_length)

    # generate extracting matrix
    # extracting_matrix = generate_extracting_matrix(weight_size, args.gfp_length, args.lfp_length)
