import numpy as np
import json
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.utils.data as data
import argparse
import logging
import os
import copy
from math import *
import random
import json

from optimizer.muon import SingleDeviceMuonWithAuxAdam
from optimizer.sophia import SophiaG


import datetime
#from torch.utils.tensorboard import SummaryWriter

from model import *
from utils import *
from vggmodel import *
from resnetcifar import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='mlp', help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='mnist', help='dataset used for training')
    parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))))
    parser.add_argument('--partition', type=str, default='homo', help='the data partitioning strategy')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--lg', default=0.1, type=float, help='learning rate')
    parser.add_argument('--epochs', type=int, default=5, help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=2,  help='number of workers in a distributed cluster')
    parser.add_argument('--alg', type=str, default='fedavg',
                            help='fl algorithms: fedavg/fedprox/scaffold/fednova/moon')
    parser.add_argument('--use_projection_head', type=bool, default=False, help='whether add an additional header to model or not (see MOON)')
    parser.add_argument('--out_dim', type=int, default=256, help='the output dimension for the projection layer')
    parser.add_argument('--loss', type=str, default='contrastive', help='for moon')
    parser.add_argument('--temperature', type=float, default=0.5, help='the temperature parameter for contrastive loss')
    parser.add_argument('--comm_round', type=int, default=50, help='number of maximum communication roun')
    parser.add_argument('--is_same_initial', type=int, default=1, help='Whether initial all the models with the same parameters in fedavg')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--dropout_p', type=float, required=False, default=0.0, help="Dropout probability. Default=0.0")
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="./models/", help='Model directory path')
    parser.add_argument('--beta', type=float, default=0.5, help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--mu', type=float, default=0.001, help='the mu parameter for fedprox')
    parser.add_argument('--noise', type=float, default=0, help='how much noise we add to some party')
    parser.add_argument('--noise_type', type=str, default='level', help='Different level of noise or different space of noise')
    parser.add_argument('--rho', type=float, default=0, help='Parameter controlling the momentum SGD')
    parser.add_argument('--sample', type=float, default=1, help='Sample ratio for each communication round')
    parser.add_argument('--K', default=1000, type=int, help='#workers')
    parser.add_argument('--alpha', default=0.5, type=float, help=' for mom_step')
    parser.add_argument('--gamma', default=0.2, type=float, help=' for mom_step')
    args = parser.parse_args()
    return args

def init_nets(net_configs, dropout_p, n_parties, args):

    nets = {net_i: None for net_i in range(n_parties)}

    if args.dataset in {'mnist', 'cifar10', 'svhn', 'fmnist'}:
        n_classes = 10
    elif args.dataset == 'celeba':
        n_classes = 2
    elif args.dataset == 'cifar100':
        n_classes = 100
    elif args.dataset == 'tinyimagenet':
        n_classes = 200
    elif args.dataset == 'femnist':
        n_classes = 62
    elif args.dataset == 'emnist':
        n_classes = 47
    elif args.dataset in {'a9a', 'covtype', 'rcv1', 'SUSY'}:
        n_classes = 2
    if args.use_projection_head:
        add = ""
        if "mnist" in args.dataset and args.model == "simple-cnn":
            add = "-mnist"
        for net_i in range(n_parties):
            net = ModelFedCon(args.model+add, args.out_dim, n_classes, net_configs)
            nets[net_i] = net
    else:
        if args.alg == 'moon':
            add = ""
            if "mnist" in args.dataset and args.model == "simple-cnn":
                add = "-mnist"
            for net_i in range(n_parties):
                net = ModelFedCon_noheader(args.model+add, args.out_dim, n_classes, net_configs)
                nets[net_i] = net
        else:
            for net_i in range(n_parties):
                if args.dataset == "generated":
                    net = PerceptronModel()
                elif args.model == "mlp":
                    if args.dataset == 'covtype':
                        input_size = 54
                        output_size = 2
                        hidden_sizes = [32,16,8]
                    elif args.dataset == 'a9a':
                        input_size = 123
                        output_size = 2
                        hidden_sizes = [32,16,8]
                    elif args.dataset == 'rcv1':
                        input_size = 47236
                        output_size = 2
                        hidden_sizes = [32,16,8]
                    elif args.dataset == 'SUSY':
                        input_size = 18
                        output_size = 2
                        hidden_sizes = [16,8]
                    net = FcNet(input_size, hidden_sizes, output_size, dropout_p)
                elif args.model == "vgg":
                    net = vgg11()
                elif args.model == "simple-cnn":
                    if args.dataset in ("cifar10", "cinic10", "svhn"):
                        net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10)
                    elif args.dataset in ("mnist", 'femnist', 'fmnist'):
                        net = SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10)
                    elif args.dataset == 'celeba':
                        net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=2)
                elif args.model == "vgg-9":
                    if args.dataset in ("mnist", 'femnist'):
                        net = ModerateCNNMNIST()
                    elif args.dataset in ("cifar10", "cinic10", "svhn"):
                        # print("in moderate cnn")
                        net = ModerateCNN()
                    elif args.dataset == 'celeba':
                        net = ModerateCNN(output_dim=2)
                elif args.model == "resnet":
                    net = ResNet50_cifar10()
                elif args.model == "vgg16":
                    net = vgg16()
                else:
                    print("not supported yet")
                    exit(1)
                nets[net_i] = net

    model_meta_data = []
    layer_type = []
    for (k, v) in nets[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)
    return nets, model_meta_data, layer_type


def train_net(net_id, net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, device="cpu"):
    logger.info('Training network %s' % str(net_id))

    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.rho, weight_decay=args.reg)
    # elif 
    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0
    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    #writer = SummaryWriter()

    for epoch in range(epochs):
        epoch_loss_collector = []
        for tmp in train_dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device)

                optimizer.zero_grad()
                x.requires_grad = True
                target.requires_grad = False
                target = target.long()

                out = net(x)
                loss = criterion(out, target)

                loss.backward()
                optimizer.step()

                cnt += 1
                epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

        #train_acc = compute_accuracy(net, train_dataloader, device=device)
        #test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

        #writer.add_scalar('Accuracy/train', train_acc, epoch)
        #writer.add_scalar('Accuracy/test', test_acc, epoch)

        # if epoch % 10 == 0:
        #     logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))
        #     train_acc = compute_accuracy(net, train_dataloader, device=device)
        #     test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)
        #
        #     logger.info('>> Training accuracy: %f' % train_acc)
        #     logger.info('>> Test accuracy: %f' % test_acc)

    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)

    # model stays on GPU — removing .to('cpu') avoids a PCIe round-trip per client per round
    logger.info(' ** Training complete **')
    return train_acc, test_acc



def train_net_fedprox(net_id, net, global_net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, mu, device="cpu"):
    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    logger.info('n_test: %d' % len(test_dataloader))

    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))


    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.rho, weight_decay=args.reg)

    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0
    # mu = 0.001
    global_weight_collector = list(global_net.to(device).parameters())

    for epoch in range(epochs):
        epoch_loss_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.to(device), target.to(device)

            optimizer.zero_grad()
            x.requires_grad = True
            target.requires_grad = False
            target = target.long()

            out = net(x)
            loss = criterion(out, target)

            #for fedprox
            fed_prox_reg = 0.0
            for param_index, param in enumerate(net.parameters()):
                fed_prox_reg += ((mu / 2) * torch.norm((param - global_weight_collector[param_index]))**2)
            loss += fed_prox_reg


            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

        # if epoch % 10 == 0:
        #     train_acc = compute_accuracy(net, train_dataloader, device=device)
        #     test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)
        #
        #     logger.info('>> Training accuracy: %f' % train_acc)
        #     logger.info('>> Test accuracy: %f' % test_acc)

    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)

    # model stays on GPU — removing .to('cpu') avoids a PCIe round-trip per client per round
    logger.info(' ** Training complete **')
    return train_acc, test_acc

def train_net_scaffold(net_id, net, global_model, c_local, c_global, train_dataloader, test_dataloader, epochs, lr, args_optimizer, device="cpu"):
    logger.info('Training network %s' % str(net_id))

    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.rho, weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0
    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    #writer = SummaryWriter()

    c_local.to(device)
    c_global.to(device)
    global_model.to(device)

    c_global_para = c_global.state_dict()
    c_local_para = c_local.state_dict()

    for epoch in range(epochs):
        epoch_loss_collector = []
        for tmp in train_dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device)

                optimizer.zero_grad()
                x.requires_grad = True
                target.requires_grad = False
                target = target.long()

                out = net(x)
                loss = criterion(out, target)

                loss.backward()
                optimizer.step()

                net_para = net.state_dict()
                for key in net_para:
                    net_para[key] = net_para[key] - args.lr * (c_global_para[key] - c_local_para[key])
                net.load_state_dict(net_para)

                cnt += 1
                epoch_loss_collector.append(loss.item())


        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

    c_new_para = c_local.state_dict()
    c_delta_para = copy.deepcopy(c_local.state_dict())
    global_model_para = global_model.state_dict()
    net_para = net.state_dict()
    for key in net_para:
        c_new_para[key] = c_new_para[key] - c_global_para[key] + (global_model_para[key] - net_para[key]) / (cnt * args.lr)
        c_delta_para[key] = c_new_para[key] - c_local_para[key]
    c_local.load_state_dict(c_new_para)


    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)

    # model stays on GPU — removing .to('cpu') avoids a PCIe round-trip per client per round
    logger.info(' ** Training complete **')
    return train_acc, test_acc, c_delta_para

def train_net_fednova(net_id, net, global_model, train_dataloader, test_dataloader, epochs, lr, args_optimizer, device="cpu"):
    logger.info('Training network %s' % str(net_id))

    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.rho, weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().to(device)

    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    #writer = SummaryWriter()


    tau = 0

    for epoch in range(epochs):
        epoch_loss_collector = []
        for tmp in train_dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device)

                optimizer.zero_grad()
                x.requires_grad = True
                target.requires_grad = False
                target = target.long()

                out = net(x)
                loss = criterion(out, target)

                loss.backward()
                optimizer.step()

                tau = tau + 1

                epoch_loss_collector.append(loss.item())


        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

    global_model.to(device)
    a_i = (tau - args.rho * (1 - pow(args.rho, tau)) / (1 - args.rho)) / (1 - args.rho)
    global_model.to(device)
    global_model_para = global_model.state_dict()
    net_para = net.state_dict()
    norm_grad = copy.deepcopy(global_model.state_dict())
    for key in norm_grad:
        #norm_grad[key] = (global_model_para[key] - net_para[key]) / a_i
        norm_grad[key] = torch.true_divide(global_model_para[key]-net_para[key], a_i)
    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)

    # model stays on GPU — removing .to('cpu') avoids a PCIe round-trip per client per round
    logger.info(' ** Training complete **')
    return train_acc, test_acc, a_i, norm_grad


def train_net_moon(net_id, net, global_net, previous_nets, train_dataloader, test_dataloader, epochs, lr, args_optimizer, mu, temperature, args,
                      round, device="cpu"):

    logger.info('Training network %s' % str(net_id))

    train_acc = compute_accuracy(net, train_dataloader, moon_model=True, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, moon_model=True, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    # conloss = ContrastiveLoss(temperature)

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)

    criterion = nn.CrossEntropyLoss().to(device)
    # global_net.to(device)

    if args.loss != 'l2norm':
        for previous_net in previous_nets:
            previous_net.to(device)
    global_w = global_net.state_dict()
    # oppsi_nets = copy.deepcopy(previous_nets)
    # for net_id, oppsi_net in enumerate(oppsi_nets):
    #     oppsi_w = oppsi_net.state_dict()
    #     prev_w = previous_nets[net_id].state_dict()
    #     for key in oppsi_w:
    #         oppsi_w[key] = 2*global_w[key] - prev_w[key]
    #     oppsi_nets.load_state_dict(oppsi_w)
    cnt = 0
    cos=torch.nn.CosineSimilarity(dim=-1).to(device)
    # mu = 0.001

    for epoch in range(epochs):
        epoch_loss_collector = []
        epoch_loss1_collector = []
        epoch_loss2_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.to(device), target.to(device)
            if target.shape[0] == 1:
                continue

            optimizer.zero_grad()
            x.requires_grad = True
            target.requires_grad = False
            target = target.long()

            _, pro1, out = net(x)
            _, pro2, _ = global_net(x)
            if args.loss == 'l2norm':
                loss2 = mu * torch.mean(torch.norm(pro2-pro1, dim=1))

            elif args.loss == 'only_contrastive' or args.loss == 'contrastive':
                posi = cos(pro1, pro2)
                logits = posi.reshape(-1,1)

                for previous_net in previous_nets:
                    previous_net.to(device)
                    _, pro3, _ = previous_net(x)
                    nega = cos(pro1, pro3)
                    logits = torch.cat((logits, nega.reshape(-1,1)), dim=1)

                    # previous_net.to('cpu')

                logits /= temperature
                labels = torch.zeros(x.size(0)).to(device).long()

                # loss = criterion(out, target) + mu * ContraLoss(pro1, pro2, pro3)

                loss2 = mu * criterion(logits, labels)

            if args.loss == 'only_contrastive':
                loss = loss2
            else:
                loss1 = criterion(out, target)
                loss = loss1 + loss2

            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())
            epoch_loss1_collector.append(loss1.item())
            epoch_loss2_collector.append(loss2.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        epoch_loss1 = sum(epoch_loss1_collector) / len(epoch_loss1_collector)
        epoch_loss2 = sum(epoch_loss2_collector) / len(epoch_loss2_collector)
        logger.info('Epoch: %d Loss: %f Loss1: %f Loss2: %f' % (epoch, epoch_loss, epoch_loss1, epoch_loss2))


    # previous_nets and current net stay on GPU — removing .to('cpu') avoids PCIe round-trips
    train_acc = compute_accuracy(net, train_dataloader, moon_model=True, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, moon_model=True, device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)
    logger.info(' ** Training complete **')
    return train_acc, test_acc

def train_net_adamw(net_id, net, global_net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, args, momen_m, momen_v, device, step):

    logger.info('Training network %s with AdamW' % str(net_id))

    train_acc = compute_accuracy(net, train_dataloader, moon_model=False, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, moon_model=False, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    # Load global weights into local model
    net.load_state_dict(global_net.state_dict())
    net.to(device)

    # Initialize momen_m if empty
    if not momen_m:
        momen_m = {k: torch.zeros_like(v) for k, v in net.state_dict().items()}
    for k in momen_m.keys():
        momen_m[k] = momen_m[k].to(device)

    # Initialize momen_v if empty
    if not momen_v:
        momen_v = {k: torch.zeros_like(v) for k, v in net.state_dict().items()}
    for k in momen_v.keys():
        momen_v[k] = momen_v[k].to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, net.parameters()),
        lr=lr,
        weight_decay=0.01,
        amsgrad=False
    )

    # Restore optimizer state from momen_v and step
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.grad is None:
                continue
            param_name = get_param_name(net, p)
            optimizer.state[p]['step'] = step.to(device)
            optimizer.state[p]['exp_avg'] = torch.zeros_like(p.data).to(device)
            optimizer.state[p]['exp_avg_sq'] = momen_v[param_name].clone().detach().to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    # Handle dataloader
    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    step = 0
    total_loss = 0

    # Training loop
    for epoch in range(epochs):
        epoch_loss_collector = []
        for tmp in train_dataloader:
            for batch_idx, (data, target) in enumerate(tmp):
                if step >= args.K:
                    break
                step += 1
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = net(data)
                loss = criterion(output, target)
                total_loss += loss.item() / args.K
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=10)
                optimizer.step()
                for n, p in net.named_parameters():
                    if not p.requires_grad:
                        continue
                    p.data.add_(momen_m[n].mul(args.gamma * lr / (args.K)))

                epoch_loss_collector.append(loss.item())

        if epoch_loss_collector:
            epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
            logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

    for group in optimizer.param_groups:
        for p in group['params']:
            if p.grad is None:
                continue
            param_name = get_param_name(net, p)
            state = optimizer.state.get(p, None)
            if state is not None and 'exp_avg_sq' in state:
                momen_v[param_name] = state['exp_avg_sq'].clone().detach().to('cpu')

    delta_w = {k: v.cpu() for k, v in net.state_dict().items()}
    for k, v in net.state_dict().items():
        delta_w[k] = v.cpu() - global_net.state_dict()[k].cpu()

    # Compute norm for logging
    norm = 0
    for k, v in net.named_parameters():
        if k in delta_w:
            norm += torch.norm(delta_w[k], p=2)

    if net_id % 10 == 0:
        logger.info('norm: %f, loss: %f' % (norm, total_loss))

    # Compute accuracy
    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)
    
    logger.info('>> Training accuracy: {}'.format(train_acc))
    logger.info('>> Test accuracy: {}'.format(test_acc))
    logger.info(' ** Training complete **')

    return train_acc, test_acc, delta_w, momen_v

def train_net_sophia(net_id, net, global_net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, args, momen_m, momen_v, device, step):
    
    logger.info('Training network %s with Sophia' % str(net_id))

    train_acc = compute_accuracy(net, train_dataloader, moon_model=False, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, moon_model=False, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))
    
    # Load global weights into local model
    net.load_state_dict(global_net.state_dict())
    net.to(device)  
    
    # Configure specific parameters for gradients if necessary
    for name, param in net.named_parameters():
        if "classifier" in name or "head" in name:
            param.requires_grad = True
            
    # Initialize momen_m if empty
    if not momen_m:
        momen_m = {k: torch.zeros_like(v) for k, v in net.state_dict().items()}
    for k in momen_m.keys():
        momen_m[k] = momen_m[k].to(device)
        
    optimizer = SophiaG(net.parameters(), lr=lr * (1 - args.gamma), betas=(0.9, 0.99), rho=0.01, weight_decay=1e-5)
    
    # Initialize optimizer state with local trackers
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.grad is None:
                continue
            param_name = get_param_name(net, p)
            if param_name in momen_v:
                optimizer.state[p]['step'] = step.to(device)
                optimizer.state[p]['exp_avg'] = torch.zeros_like(p.data).to(device)
                optimizer.state[p]['hessian'] = momen_v[param_name].clone().detach().to(device)
                
    criterion = nn.CrossEntropyLoss().to(device)
    
    # Handle dataloader
    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]
        
    step = 0
    total_loss = 0
    
    # Training loop
    for epoch in range(epochs):
        epoch_loss_collector = []
        for tmp in train_dataloader:
            for batch_idx, (data, target) in enumerate(tmp):
                if step >= args.K:
                    break
                step += 1
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = net(data)
                loss = criterion(output, target)
                total_loss += loss.item() / args.K
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=10)
                optimizer.step()
                
                for n, p in net.named_parameters():
                    if not p.requires_grad:
                        continue
                    p.data.add_(momen_m[n].mul(args.gamma * lr / (args.K)))
                    
                epoch_loss_collector.append(loss.item())
                
        if epoch_loss_collector:
            epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
            logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))
            
    # Extract structural hessian state back to CPU tracking dictionary
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.grad is None:
                continue
            param_name = get_param_name(net, p)
            state = optimizer.state.get(p, None)
            if state is not None and 'hessian' in state:
                momen_v[param_name] = state['hessian'].clone().detach().to('cpu')
                
    # Compute weight updates delta_w
    delta_w = {}
    for k, v in net.state_dict().items():
        delta_w[k] = v.cpu() - global_net.state_dict()[k].cpu()
        
    # Compute norm for logging metrics
    norm = 0
    for k, v in net.named_parameters():
        if k in delta_w:
            norm += torch.norm(delta_w[k], p=2)
            
    # if net_id % 10 == 0:
    #     logger.info('norm: %f, loss: %f' % (norm, total_loss))
        
    # Post-training evaluation
    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)
    
    logger.info('>> Training accuracy: {}'.format(train_acc))
    logger.info('>> Test accuracy: {}'.format(test_acc))
    logger.info(' ** Training complete **')

    return train_acc, test_acc, delta_w, momen_v

def train_net_muon(net_id, net, global_net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, args, ps_c, momen_m, device, step):
    
    logger.info('Training network %s with Muon' % str(net_id))

    train_acc = compute_accuracy(net, train_dataloader, moon_model=False, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, moon_model=False, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))
    
    # Load global weights into local model
    net.load_state_dict(global_net.state_dict())
    net.to(device)
    
    # Initialize ps_c if empty
    if not ps_c:
        ps_c = {k: torch.zeros_like(v) for k, v in net.state_dict().items()}
    for k in ps_c.keys():
        ps_c[k] = ps_c[k].to(device)
    
    # Initialize momen_m if empty
    if not momen_m:
        momen_m = {k: torch.zeros_like(v) for k, v in net.state_dict().items()}
    for k in momen_m.keys():
        momen_m[k] = momen_m[k].to(device)
    
    # Separate parameters for Muon and Adam
    hidden_weights = [p for p in net.parameters() if p.ndim >= 2]
    hidden_gains_biases = [p for p in net.parameters() if p.ndim < 2]
    nonhidden_params = []
    
    param_groups = [
        dict(params=hidden_weights, use_muon=True,
             lr=lr * args.alpha * (1 - args.gamma), weight_decay=0.01, momentum=0.95),
        dict(params=hidden_gains_biases + nonhidden_params, use_muon=False,
             lr=lr * (1 - args.gamma), betas=(0.9, 0.95), weight_decay=0.01),
    ]
    
    optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.grad is None:
                continue
            param_name = get_param_name(net, p)
            if p.ndim >= 2 and param_name in momen_m:
                optimizer.state[p]['momentum_buffer'] = momen_m[param_name].clone().detach().to(device)
    
    criterion = nn.CrossEntropyLoss().to(device)
    
    # Handle dataloader
    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]
    
    step = 0
    total_loss = 0
    
    # Training loop
    for epoch in range(epochs):
        epoch_loss_collector = []
        for tmp in train_dataloader:
            for batch_idx, (data, target) in enumerate(tmp):
                if step >= args.K:
                    break
                step += 1
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = net(data)
                loss = criterion(output, target)
                total_loss += loss.item() / args.K
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=1)
                optimizer.step()
                for n, p in net.named_parameters():
                    if not p.requires_grad:
                        continue
                    p.data.add_(ps_c[n].mul(args.gamma * lr / (args.K)))
                
                epoch_loss_collector.append(loss.item())
        
        if epoch_loss_collector:
            epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
            logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))
    
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.grad is None:
                continue
            param_name = get_param_name(net, p)
            state = optimizer.state.get(p, None)
            if p.ndim >= 2 and state is not None and 'momentum_buffer' in state:
                momen_m[param_name] = state['momentum_buffer'].clone().detach().to('cpu')
    
    delta_w = {k: v.cpu() for k, v in net.state_dict().items()}
    for k, v in net.state_dict().items():
        delta_w[k] = v.cpu() - global_net.state_dict()[k].cpu()
    
    # Compute norm for logging
    norm = 0
    for k, v in net.named_parameters():
        if k in delta_w:
            norm += torch.norm(delta_w[k], p=2)
    
    # if net_id % 10 == 0:
    #     logger.info('norm: %f, loss: %f' % (norm, total_loss))
    
    # Compute accuracy
    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)
    
    logger.info('>> Training accuracy: {}'.format(train_acc))
    logger.info('>> Test accuracy: {}'.format(test_acc))
    logger.info(' ** Training complete **')
    
    return train_acc, test_acc, delta_w, momen_m


def get_param_name(net, param):
    """Helper function to get parameter name from parameter object"""
    for name, p in net.named_parameters():
        if p is param:
            return name
    return None
    

def view_image(train_dataloader):
    for (x, target) in train_dataloader:
        np.save("img.npy", x)
        print(x.shape)
        exit(0)


def local_train_net(nets, selected, args, net_dataidx_map, test_dl=None, device="cpu", train_dl_cache=None):
    avg_acc = 0.0

    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]

        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        net.to(device)

        if train_dl_cache is not None:
            # reuse the pre-built DataLoader — avoids recreating it every round
            train_dl_local = train_dl_cache[net_id]
        else:
            noise_level = args.noise
            if net_id == args.n_parties - 1:
                noise_level = 0
            if args.noise_type == 'space':
                train_dl_local, _, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level, net_id, args.n_parties-1)
            else:
                noise_level = args.noise / (args.n_parties - 1) * net_id
                train_dl_local, _, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level)
        n_epoch = args.epochs


        trainacc, testacc = train_net(net_id, net, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, device=device)
        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc
        # saving the trained models here
        # save_model(net, net_id, args)
        # else:
        #     load_model(net, net_id, device=device)
    avg_acc /= len(selected)
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)

    nets_list = list(nets.values())
    return nets_list


def local_train_net_fedprox(nets, selected, global_model, args, net_dataidx_map, test_dl=None, device="cpu", train_dl_cache=None):
    avg_acc = 0.0

    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]

        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        net.to(device)

        if train_dl_cache is not None:
            # reuse the pre-built DataLoader — avoids recreating it every round
            train_dl_local = train_dl_cache[net_id]
        else:
            noise_level = args.noise
            if net_id == args.n_parties - 1:
                noise_level = 0
            if args.noise_type == 'space':
                train_dl_local, _, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level, net_id, args.n_parties-1)
            else:
                noise_level = args.noise / (args.n_parties - 1) * net_id
                train_dl_local, _, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level)
        n_epoch = args.epochs

        trainacc, testacc = train_net_fedprox(net_id, net, global_model, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, args.mu, device=device)
        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc
    avg_acc /= len(selected)
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)

    nets_list = list(nets.values())
    return nets_list

def local_train_net_scaffold(nets, selected, global_model, c_nets, c_global, args, net_dataidx_map, test_dl=None, device="cpu", train_dl_cache=None):
    avg_acc = 0.0

    total_delta = copy.deepcopy(global_model.state_dict())
    for key in total_delta:
        total_delta[key] = 0.0
    c_global.to(device)
    global_model.to(device)
    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]

        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        net.to(device)
        c_nets[net_id].to(device)

        if train_dl_cache is not None:
            # reuse the pre-built DataLoader — avoids recreating it every round
            train_dl_local = train_dl_cache[net_id]
        else:
            noise_level = args.noise
            if net_id == args.n_parties - 1:
                noise_level = 0
            if args.noise_type == 'space':
                train_dl_local, _, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level, net_id, args.n_parties-1)
            else:
                noise_level = args.noise / (args.n_parties - 1) * net_id
                train_dl_local, _, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level)
        n_epoch = args.epochs

        trainacc, testacc, c_delta_para = train_net_scaffold(net_id, net, global_model, c_nets[net_id], c_global, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, device=device)

        # c_net stays on GPU — removing .to('cpu') avoids a PCIe round-trip per client per round
        for key in total_delta:
            total_delta[key] += c_delta_para[key]


        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc
    for key in total_delta:
        total_delta[key] /= args.n_parties
    c_global_para = c_global.state_dict()
    for key in c_global_para:
        if c_global_para[key].type() == 'torch.LongTensor':
            c_global_para[key] += total_delta[key].type(torch.LongTensor)
        elif c_global_para[key].type() == 'torch.cuda.LongTensor':
            c_global_para[key] += total_delta[key].type(torch.cuda.LongTensor)
        else:
            #print(c_global_para[key].type())
            c_global_para[key] += total_delta[key]
    c_global.load_state_dict(c_global_para)

    avg_acc /= len(selected)
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)

    nets_list = list(nets.values())
    return nets_list

def local_train_net_precond(nets, selected, global_model, args, net_dataidx_map, ps_c, v, alg, test_dl=None, device="cpu", train_dl_cache=None, step=None):
    avg_acc = 0.0

    m_list = []
    d_list = []
    n_list = []
    global_model.to(device)
    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]

        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        net.to(device)

        if train_dl_cache is not None:
            # reuse the pre-built DataLoader — avoids recreating it every round
            train_dl_local = train_dl_cache[net_id]
        else:
            noise_level = args.noise
            if net_id == args.n_parties - 1:
                noise_level = 0
            if args.noise_type == 'space':
                train_dl_local, _, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level, net_id, args.n_parties-1)
            else:
                noise_level = args.noise / (args.n_parties - 1) * net_id
                train_dl_local, _, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level)
        n_epoch = args.epochs

        if alg == 'fedmuon':
            trainacc, testacc, d_i, m_i = train_net_muon(net_id, net, global_model, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, args, ps_c, v, device, step)
        elif alg == 'fedadamw':
            trainacc, testacc, d_i, m_i = train_net_adamw(net_id, net, global_model, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, args, ps_c, v, device, step)
        elif alg == 'fedsophia':
            trainacc, testacc, d_i, m_i = train_net_sophia(net_id, net, global_model, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, args, ps_c, v, device, step)
        m_list.append(m_i)
        d_list.append(d_i)
        n_i = len(train_dl_local.dataset)
        n_list.append(n_i)
        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc


    avg_acc /= len(selected)
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)

    nets_list = list(nets.values())
    return nets_list, m_list, d_list, n_list


def local_train_net_fednova(nets, selected, global_model, args, net_dataidx_map, test_dl=None, device="cpu", train_dl_cache=None):
    avg_acc = 0.0

    a_list = []
    d_list = []
    n_list = []
    global_model.to(device)
    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]

        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        net.to(device)

        if train_dl_cache is not None:
            # reuse the pre-built DataLoader — avoids recreating it every round
            train_dl_local = train_dl_cache[net_id]
        else:
            noise_level = args.noise
            if net_id == args.n_parties - 1:
                noise_level = 0
            if args.noise_type == 'space':
                train_dl_local, _, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level, net_id, args.n_parties-1)
            else:
                noise_level = args.noise / (args.n_parties - 1) * net_id
                train_dl_local, _, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level)
        n_epoch = args.epochs


        trainacc, testacc, a_i, d_i = train_net_fednova(net_id, net, global_model, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, device=device)

        a_list.append(a_i)
        d_list.append(d_i)
        n_i = len(train_dl_local.dataset)
        n_list.append(n_i)
        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc


    avg_acc /= len(selected)
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)

    nets_list = list(nets.values())
    return nets_list, a_list, d_list, n_list

def local_train_net_moon(nets, selected, args, net_dataidx_map, test_dl=None, global_model=None, prev_model_pool=None, round=None, device="cpu", train_dl_cache=None):
    avg_acc = 0.0
    global_model.to(device)
    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]

        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        net.to(device)

        if train_dl_cache is not None:
            # reuse the pre-built DataLoader — avoids recreating it every round
            train_dl_local = train_dl_cache[net_id]
        else:
            noise_level = args.noise
            if net_id == args.n_parties - 1:
                noise_level = 0
            if args.noise_type == 'space':
                train_dl_local, _, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level, net_id, args.n_parties-1)
            else:
                noise_level = args.noise / (args.n_parties - 1) * net_id
                train_dl_local, _, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level)
        n_epoch = args.epochs

        prev_models=[]
        for i in range(len(prev_model_pool)):
            prev_models.append(prev_model_pool[i][net_id])
        trainacc, testacc = train_net_moon(net_id, net, global_model, prev_models, train_dl_local, test_dl, n_epoch, args.lr,
                                              args.optimizer, args.mu, args.temperature, args, round, device=device)
        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc

    avg_acc /= len(selected)
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)
    # global_model stays on GPU — removing .to('cpu') avoids a PCIe round-trip per round
    nets_list = list(nets.values())
    return nets_list



def get_partition_dict(dataset, partition, n_parties, init_seed=0, datadir='./data', logdir='./logs', beta=0.5):
    seed = init_seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        dataset, datadir, logdir, partition, n_parties, beta=beta)

    return net_dataidx_map

if __name__ == '__main__':
    # torch.set_printoptions(profile="full")
    step = torch.tensor([0], dtype=torch.float32, device='cpu')
    args = get_args()
    mkdirs(args.logdir)
    mkdirs(args.modeldir)
    experiment_path = f"{args.model}/{args.dataset}/{args.partition}"
    time_id = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S")
    if args.log_file_name is None:
        argument_path='%s_arguments-%s.json' % (args.alg, time_id)
    else:
        argument_path=args.log_file_name+'.json'

    os.makedirs(os.path.join(args.logdir, experiment_path), exist_ok=True)

    argument_path = os.path.join(args.logdir, experiment_path, argument_path)
    with open(argument_path, 'w') as f:
        json.dump(str(args), f)
    device = torch.device(args.device)
    # logging.basicConfig(filename='test.log', level=logger.info, filemode='w')
    # logging.info("test")
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    if args.log_file_name is None:
        args.log_file_name = '%s_log-%s' % (args.alg, time_id)
    log_path=args.log_file_name+'.log'
    logging.basicConfig(
        filename=os.path.join(args.logdir, experiment_path, log_path),
        # filename='/home/qinbin/test.log',
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.info(device)


    seed = args.init_seed
    logger.info("#" * 100)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    logger.info("Partitioning data")
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        args.dataset, args.datadir, args.logdir, args.partition, args.n_parties, beta=args.beta)

    n_classes = len(np.unique(y_train))

    train_dl_global, test_dl_global, train_ds_global, test_ds_global = get_dataloader(args.dataset,
                                                                                        args.datadir,
                                                                                        args.batch_size,
                                                                                        32)

    print("len train_dl_global:", len(train_ds_global))

    data_size = len(test_ds_global)

    # Pre-load the full test set into GPU memory once — all eval passes reuse these tensors
    # with zero PCIe transfers (x.to(device) in compute_accuracy becomes a no-op)
    test_dl_global = build_gpu_test_loader(args.dataset, args.datadir, 32, device)

    # Pre-build one DataLoader per client — created once, reused every communication round
    # This eliminates n_parties * comm_round DataLoader instantiations (500+ for this run)
    client_train_loaders = {}
    for net_id in range(args.n_parties):
        dataidxs = net_dataidx_map[net_id]
        noise_level = args.noise
        if net_id == args.n_parties - 1:
            noise_level = 0
        if args.noise_type == 'space':
            dl, _, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level, net_id, args.n_parties-1)
        else:
            noise_level = args.noise / (args.n_parties - 1) * net_id
            dl, _, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level)
        client_train_loaders[net_id] = dl

    train_all_in_list = []
    test_all_in_list = []
    if args.noise > 0:
        for party_id in range(args.n_parties):
            dataidxs = net_dataidx_map[party_id]

            noise_level = args.noise
            if party_id == args.n_parties - 1:
                noise_level = 0

            if args.noise_type == 'space':
                train_dl_local, test_dl_local, train_ds_local, test_ds_local = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level, party_id, args.n_parties-1)
            else:
                noise_level = args.noise / (args.n_parties - 1) * party_id
                train_dl_local, test_dl_local, train_ds_local, test_ds_local = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level)
            train_all_in_list.append(train_ds_local)
            test_all_in_list.append(test_ds_local)
        train_all_in_ds = data.ConcatDataset(train_all_in_list)
        train_dl_global = data.DataLoader(dataset=train_all_in_ds, batch_size=args.batch_size, shuffle=True)
        test_all_in_ds = data.ConcatDataset(test_all_in_list)
        test_dl_global = data.DataLoader(dataset=test_all_in_ds, batch_size=32, shuffle=False)



    if args.alg == 'fedavg':
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        global_para = global_model.state_dict()
        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        for round in range(args.comm_round):
            logger.info("in comm round:" + str(round))

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)

            local_train_net(nets, selected, args, net_dataidx_map, test_dl=test_dl_global, device=device, train_dl_cache=client_train_loaders)

            # aggregate on GPU — .state_dict() keeps tensors on their current device,
            # so no .cpu() call needed and arithmetic stays on the GPU
            total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]

            for idx in range(len(selected)):
                net_para = nets[selected[idx]].state_dict()
                if idx == 0:
                    for key in net_para:
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]
            global_model.load_state_dict(global_para)

            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl_global))

            # global_model already on GPU — .to(device) removed
            train_acc = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc, conf_matrix = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True, device=device)

            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)


    elif args.alg == 'fedprox':
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        global_para = global_model.state_dict()

        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        for round in range(args.comm_round):
            logger.info("in comm round:" + str(round))

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)

            local_train_net_fedprox(nets, selected, global_model, args, net_dataidx_map, test_dl=test_dl_global, device=device, train_dl_cache=client_train_loaders)
            # global_model stays on GPU — removed .to('cpu') that was placed here before aggregation

            # aggregate on GPU — .state_dict() keeps tensors on their current device,
            # so no .cpu() call needed and arithmetic stays on the GPU
            total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]

            for idx in range(len(selected)):
                net_para = nets[selected[idx]].state_dict()
                if idx == 0:
                    for key in net_para:
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]
            global_model.load_state_dict(global_para)

            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl_global))

            # global_model already on GPU — .to(device) removed
            train_acc = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc, conf_matrix = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True, device=device)

            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)

    elif args.alg == 'scaffold':
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        c_nets, _, _ = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        c_globals, _, _ = init_nets(args.net_config, 0, 1, args)
        c_global = c_globals[0]
        c_global_para = c_global.state_dict()
        for net_id, net in c_nets.items():
            net.load_state_dict(c_global_para)

        global_para = global_model.state_dict()
        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)


        for round in range(args.comm_round):
            logger.info("in comm round:" + str(round))

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)

            local_train_net_scaffold(nets, selected, global_model, c_nets, c_global, args, net_dataidx_map, test_dl=test_dl_global, device=device, train_dl_cache=client_train_loaders)

            # aggregate on GPU — .state_dict() keeps tensors on their current device,
            # so no .cpu() call needed and arithmetic stays on the GPU
            total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]

            for idx in range(len(selected)):
                net_para = nets[selected[idx]].state_dict()
                if idx == 0:
                    for key in net_para:
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]
            global_model.load_state_dict(global_para)

            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl_global))

            # global_model already on GPU — .to(device) removed
            train_acc = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc, conf_matrix = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True, device=device)

            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)

    elif args.alg in {'fedmuon', 'fedadamw', 'fedsophia'}:
        logger.info("Initializing nets for FedMuon")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        d_list = [copy.deepcopy(global_model.state_dict()) for i in range(args.n_parties)]
        d_total_round = copy.deepcopy(global_model.state_dict())
        for i in range(args.n_parties):
            for key in d_list[i]:
                d_list[i][key] = 0
        for key in d_total_round:
            d_total_round[key] = 0

        m = {} 
        v = {}

        for round in range(args.comm_round):
            logger.info("in comm round:" + str(round))

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)

            # 1. Run local training
            _, m_list, d_list, n_list = local_train_net_precond(
                nets, selected, global_model, args, net_dataidx_map, 
                ps_c=m, v=v, alg=args.alg, test_dl=test_dl_global, device=device, 
                train_dl_cache=client_train_loaders, step = step
            )
            
            # Number of active clients in this round
            n_selected = len(selected)
            
            # 2. Aggregate client momentum vectors (momen_v)
            momen_v = {}
            if len(m_list) > 0:
                for k, val in m_list[0].items():
                    momen_v[k] = val / n_selected

                for ci in m_list[1:]:
                    for k, val in ci.items():
                        momen_v[k] += val / n_selected
            
            # Update the global tracking dictionary v with the aggregated momentum
            v.update(momen_v)

            # 3. Aggregate model updates / weights from d_list_selected
            sum_weights = {}
            if len(d_list) > 0:
                for k, val in d_list[0].items():
                    sum_weights[k] = val / n_selected
                    
                for weight in d_list[1:]:
                    for k, val in weight.items():
                        sum_weights[k] += val / n_selected

            # 4. Update the global optimizer state tracker (m) using sum_weights
            for k, val in sum_weights.items():
                if k not in m.keys():
                    m[k] = sum_weights[k] / args.lr
                else:
                    m[k] = sum_weights[k] / args.lr
                    # m[k] = args.alpha * m[k] + (1 - args.alpha) * sum_weights[k] / args.lr # alternative momentum tracking

            # 5. Apply aggregated updates to the global model parameters
            global_model.to('cpu')
            ps_w = global_model.state_dict()
            for k in sum_weights.keys():
                ps_w[k] = ps_w[k] + sum_weights[k]
            
            global_model.load_state_dict(ps_w)
            global_model.to(device)

            # 6. Global Model Tracking and Verification
            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl_global))

            train_acc = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc, conf_matrix = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True, device=device)

            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)

    elif args.alg == 'fednova':
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        d_list = [copy.deepcopy(global_model.state_dict()) for i in range(args.n_parties)]
        d_total_round = copy.deepcopy(global_model.state_dict())
        for i in range(args.n_parties):
            for key in d_list[i]:
                d_list[i][key] = 0
        for key in d_total_round:
            d_total_round[key] = 0

        data_sum = 0
        for i in range(args.n_parties):
            data_sum += len(traindata_cls_counts[i])
        portion = []
        for i in range(args.n_parties):
            portion.append(len(traindata_cls_counts[i]) / data_sum)

        global_para = global_model.state_dict()
        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        for round in range(args.comm_round):
            logger.info("in comm round:" + str(round))

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)

            _, a_list, d_list, n_list = local_train_net_fednova(nets, selected, global_model, args, net_dataidx_map, test_dl=test_dl_global, device=device, train_dl_cache=client_train_loaders)
            total_n = sum(n_list)
            # d_total_round stays on GPU — state_dict tensors are on the same device as global_model
            d_total_round = copy.deepcopy(global_model.state_dict())
            for key in d_total_round:
                d_total_round[key] = 0.0

            for i in range(len(selected)):
                d_para = d_list[i]
                for key in d_para:
                    d_total_round[key] += d_para[key] * n_list[i] / total_n

            # update global model — all arithmetic on GPU
            coeff = 0.0
            for i in range(len(selected)):
                coeff = coeff + a_list[i] * n_list[i]/total_n

            updated_model = global_model.state_dict()
            for key in updated_model:
                if updated_model[key].type() == 'torch.LongTensor':
                    updated_model[key] -= (coeff * d_total_round[key]).type(torch.LongTensor)
                elif updated_model[key].type() == 'torch.cuda.LongTensor':
                    updated_model[key] -= (coeff * d_total_round[key]).type(torch.cuda.LongTensor)
                else:
                    updated_model[key] -= coeff * d_total_round[key]
            global_model.load_state_dict(updated_model)

            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl_global))

            # global_model already on GPU — .to(device) removed
            train_acc = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc, conf_matrix = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True, device=device)

            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)

    elif args.alg == 'moon':
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        global_para = global_model.state_dict()
        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        old_nets_pool = []
        old_nets = copy.deepcopy(nets)
        for _, net in old_nets.items():
            net.eval()
            for param in net.parameters():
                param.requires_grad = False

        for round in range(args.comm_round):
            logger.info("in comm round:" + str(round))

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)

            local_train_net_moon(nets, selected, args, net_dataidx_map, test_dl=test_dl_global, global_model=global_model,
                                 prev_model_pool=old_nets_pool, round=round, device=device, train_dl_cache=client_train_loaders)

            # aggregate on GPU — .state_dict() keeps tensors on their current device,
            # so no .cpu() call needed and arithmetic stays on the GPU
            total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]

            for idx in range(len(selected)):
                net_para = nets[selected[idx]].state_dict()
                if idx == 0:
                    for key in net_para:
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]
            global_model.load_state_dict(global_para)

            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl_global))

            # global_model already on GPU — .to(device) removed
            train_acc = compute_accuracy(global_model, train_dl_global, moon_model=True, device=device)
            test_acc, conf_matrix = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True, moon_model=True, device=device)

            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)

            old_nets = copy.deepcopy(nets)
            for _, net in old_nets.items():
                net.eval()
                for param in net.parameters():
                    param.requires_grad = False
            if len(old_nets_pool) < 1:
                old_nets_pool.append(old_nets)
            else:
                old_nets_pool[0] = old_nets

    elif args.alg == 'local_training':
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        arr = np.arange(args.n_parties)
        local_train_net(nets, arr, args, net_dataidx_map, test_dl=test_dl_global, device=device, train_dl_cache=client_train_loaders)

    elif args.alg == 'all_in':
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, 1, args)
        n_epoch = args.epochs
        nets[0].to(device)
        trainacc, testacc = train_net(0, nets[0], train_dl_global, test_dl_global, n_epoch, args.lr, args.optimizer, device=device)

        logger.info("All in test acc: %f" % testacc)

   
