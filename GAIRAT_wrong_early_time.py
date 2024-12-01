#===============
import shutil
#===============
import os
import argparse
import torchvision
import torch.optim as optim
from torchvision import transforms
from models import *
from GAIR import GAIR
import numpy as np
import attack_generator as attack
from utils import Logger
import time
parser = argparse.ArgumentParser(description='GAIRAT: Geometry-aware instance-dependent adversarial training')
#==============================================================
# 4.13 1:15
parser.add_argument('--w-logits-base1', default=0.5, type=float)
parser.add_argument('--w-logits-base2', default=0.3, type=float)
parser.add_argument('--w-logits-base3', default=0.1, type=float)
parser.add_argument('--w-logits-base4', default=0, type=float)
parser.add_argument('--begin-epoch-2', default=0, type=float)
parser.add_argument('--begin-epoch-3', default=0, type=float)
parser.add_argument('--begin-epoch-4', default=0, type=float)
parser.add_argument('--classify_step', type=int, default=5)
parser.add_argument('--gair', default=False, action="store_true")
# parser.add_argument('--soft', default=False, action="store_true")
# parser.add_argument('--way', default='logits', type=str)
# parser.add_argument('--p', default=1, type=int)
#==============================================================
parser.add_argument('--epochs', type=int, default=120, metavar='N', help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=2e-4, type=float, metavar='W')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum')
parser.add_argument('--epsilon', type=float, default=0.031, help='perturbation bound')
parser.add_argument('--num-steps', type=int, default=10, help='maximum perturbation step K')
parser.add_argument('--step-size', type=float, default=0.007, help='step size')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')
parser.add_argument('--net', type=str, default="WRN",help="decide which network to use,choose from smallcnn,resnet18,WRN")
parser.add_argument('--dataset', type=str, default="cifar10", help="choose from cifar10,svhn,cifar100,mnist")
parser.add_argument('--random',type=bool,default=True,help="whether to initiat adversarial sample with random noise")
parser.add_argument('--depth',type=int,default=32,help='WRN depth')
parser.add_argument('--width-factor',type=int,default=10,help='WRN width factor')
parser.add_argument('--drop-rate',type=float,default=0.0, help='WRN drop rate')
parser.add_argument('--resume',type=str,default=None,help='whether to resume training')
parser.add_argument('--out-dir',type=str,default='./GAIRAT_result',help='dir of output')
parser.add_argument('--lr-schedule', default='piecewise', choices=['superconverge', 'piecewise', 'linear', 'onedrop', 'multipledecay', 'cosine'])
parser.add_argument('--lr-max', default=0.1, type=float)
parser.add_argument('--lr-one-drop', default=0.01, type=float)
parser.add_argument('--lr-drop-epoch', default=100, type=int)
parser.add_argument('--Lambda',type=str, default='-1.0', help='parameter for GAIR')
parser.add_argument('--Lambda_max',type=float, default=float('inf'), help='max Lambda')
parser.add_argument('--Lambda_schedule', default='fixed', choices=['linear', 'piecewise', 'fixed'])
parser.add_argument('--weight_assignment_function', default='Tanh', choices=['Discrete','Sigmoid','Tanh'])
parser.add_argument('--begin_epoch', type=int, default=60, help='when to use GAIR')
args = parser.parse_args()

# Training settings
seed = args.seed
momentum = args.momentum
weight_decay = args.weight_decay
depth = args.depth
width_factor = args.width_factor
drop_rate = args.drop_rate
resume = args.resume
out_dir = args.out_dir

torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

# Models and optimizer
if args.net == "smallcnn":
    model = SmallCNN().cuda()
    net = "smallcnn"
if args.net == "resnet18":
    model = ResNet18().cuda()
    net = "resnet18"
if args.net == "preactresnet18":
    model = PreActResNet18().cuda()
    net = "preactresnet18"
if args.net == "WRN":
    model = Wide_ResNet_Madry(depth=depth, num_classes=10, widen_factor=width_factor, dropRate=drop_rate).cuda()
    net = "WRN{}-{}-dropout{}".format(depth,width_factor,drop_rate)

model = torch.nn.DataParallel(model)
optimizer = optim.SGD(model.parameters(), lr=args.lr_max, momentum=momentum, weight_decay=weight_decay)

# Learning schedules
if args.lr_schedule == 'superconverge':
    lr_schedule = lambda t: np.interp([t], [0, args.epochs * 2 // 5, args.epochs], [0, args.lr_max, 0])[0]
elif args.lr_schedule == 'piecewise':
    def lr_schedule(t):
        if args.epochs >= 110:
            # Train Wide-ResNet
            if t / args.epochs < 0.5:
                return args.lr_max
            elif t / args.epochs < 0.75:
                return args.lr_max / 10.
            elif t / args.epochs < (11/12):
                return args.lr_max / 100.
            else:
                return args.lr_max / 200.
        else:
            # Train ResNet
            if t / args.epochs < 0.3:
                return args.lr_max
            elif t / args.epochs < 0.6:
                return args.lr_max / 10.
            else:
                return args.lr_max / 100.
elif args.lr_schedule == 'linear':
    lr_schedule = lambda t: np.interp([t], [0, args.epochs // 3, args.epochs * 2 // 3, args.epochs], [args.lr_max, args.lr_max, args.lr_max / 10, args.lr_max / 100])[0]
elif args.lr_schedule == 'onedrop':
    def lr_schedule(t):
        if t < args.lr_drop_epoch:
            return args.lr_max
        else:
            return args.lr_one_drop
elif args.lr_schedule == 'multipledecay':
    def lr_schedule(t):
        return args.lr_max - (t//(args.epochs//10))*(args.lr_max/10)
elif args.lr_schedule == 'cosine': 
    def lr_schedule(t): 
        return args.lr_max * 0.5 * (1 + np.cos(t / args.epochs * np.pi))


# Store path
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

#======================================
# 4.17 13:01
cifar10_mean = (0.4914, 0.4822, 0.4465) # equals np.mean(train_set.train_data, axis=(0,1,2))/255
cifar10_std = (0.2471, 0.2435, 0.2616) # equals np.std(train_set.train_data, axis=(0,1,2))/255
mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
std = torch.tensor(cifar10_std).view(3,1,1).cuda()

def normalize(X):
    if args.dataset == 'cifar10':
        return (X - mu)/std
    else:
        return X
#     return X
#======================================

# Save checkpoint
def save_checkpoint(state, checkpoint=out_dir, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

# Get adversarially robust network
def train(epoch, model, train_loader, optimizer, Lambda):
    
    lr = 0
    num_data = 0
    train_robust_loss = 0

    #==============================
    # 4.28 12:29
    # delta_logits_all = torch.zeros(50000, args.num_steps + 1, 10)
    # correct_clean_all = torch.zeros(50000)
    # correct_early_all = torch.zeros(50000)
    # label_true = torch.zeros(50000)
    #==============================
    
    # time_s = time.time()
    # time_s2 = time.time()

    for batch_idx, (data, target) in enumerate(train_loader):
        # print('time:')
        # print(time.time() - time_s)
        
        # time_s = time.time()

        loss = 0
        data, target = data.cuda(), target.cuda()
        # Get adversarial data and geometry value
        #====================================================
        # 4.16 23:17
        # x_adv, Kappa, logits_margin = attack.GA_PGD(model,data,target,args.epsilon,args.step_size,args.num_steps,loss_fn="cent",category="Madry",rand_init=True)
        # x_adv, Kappa, logits_margin = attack.GA_PGD_margin(model,data,target,args.epsilon,args.step_size,args.num_steps,loss_fn="cent",category="Madry",rand_init=True)
        # x_adv, Kappa, w_logits, delta_logits, correct_clean, correct_early = attack.GA_PGD_early(model,data,target,args.epsilon,args.step_size,args.num_steps,loss_fn="cent",category="Madry",rand_init=True,classify_step=args.classify_step)
        if (epoch + 1) >= args.begin_epoch:
            if args.gair:
                x_adv, Kappa = attack.GA_PGD(model,data,target,args.epsilon,args.step_size,args.num_steps,loss_fn="cent",category="Madry",rand_init=True)
                w_logits = torch.zeros(2)
            else:
                x_adv, w_logits = attack.GA_PGD_early2(model,data,target,args.epsilon,args.step_size,args.num_steps,loss_fn="cent",category="Madry",rand_init=True,classify_step=args.classify_step)
        else:
            x_adv, Kappa = attack.GA_PGD(model,data,target,args.epsilon,args.step_size,args.num_steps,loss_fn="cent",category="Madry",rand_init=True)
        
        # delta_logits_all[batch_idx * 128 : batch_idx * 128 + len(data)] = delta_logits
        # correct_clean_all[batch_idx * 128 : batch_idx * 128 + len(data)] = correct_clean
        # correct_early_all[batch_idx * 128 : batch_idx * 128 + len(data)] = correct_early
        # label_true[batch_idx * 128 : batch_idx * 128 + len(data)] = target
        #====================================================
        # logits_nat = model(normalize(data)).detach().clone()
        # logits_adv = model(normalize(x_adv)).detach().clone()
        model.train()
        lr = lr_schedule(epoch + 1)
        optimizer.param_groups[0].update(lr=lr)
        optimizer.zero_grad()
        #===============================
        # 4.17 21:58
#         logit = model(x_adv)
        logit = model(normalize(x_adv))
        #===============================
        
        #---------------------------------
        # 4.13 0:42
        if (epoch + 1) >= args.begin_epoch:
            if not args.gair:
                target = target[torch.where(w_logits == 1)].detach()
            #===============
#             logits_nat = model(data).detach().clone()
#             logits_adv = model(x_adv).detach().clone()
            
            #===============
#             if args.soft:
#                 softmax = nn.Softmax(dim=1)
#             else:
#                 def softmax(xx):
#                     return xx
#             if args.way == 'logits':
#                 logits_final = logits_nat
#             else:
#                 logits_final = logits_margin
            
            
            
            # pred = logits_adv.max(1, keepdim=True)[1]
            # correct = pred.eq(target.view_as(pred)).float().squeeze(dim=1)
            # w_logits = torch.zeros_like(correct).cuda()
            # w_logits[torch.where(correct == 0)] = 1
            # w_logits[torch.where(correct == 1)] = 0
            
            
            
#             w_logits = torch.norm(softmax(logits_final) - softmax(logits_adv), p=args.p, dim=1).detach().clone()
#             w_logits = nn.MSELoss(reduce=False)(logits_margin, logits_adv).mean(dim = 1).detach().clone()
            
#             if epoch < args.begin_epoch_2:
#                 w_logits_base = args.w_logits_base1
#             elif epoch < args.begin_epoch_3:
#                 w_logits_base = args.w_logits_base2
#             elif epoch < args.begin_epoch_4:
#                 w_logits_base = args.w_logits_base3
#             else:
#                 w_logits_base = args.w_logits_base4

#             # 预归一化
#             w_logits = w_logits * len(w_logits) / w_logits.sum()
#             # 减小方差
#             w_logits = w_logits + w_logits_base
            
#             LAMBDA = -1.0
#             NUM_STEPS = w_logits.max()

#             w_logits = 1 - (torch.tanh(LAMBDA+((NUM_STEPS/2)-w_logits)*5/((NUM_STEPS/2)))+1)/2
            # 归一化
            if w_logits.sum() == 0:
                normalized_reweight = torch.zeros(len(x_adv)).cuda()
            else:
                # normalized_reweight = w_logits * len(w_logits) / w_logits.sum()
                normalized_reweight = torch.ones(len(x_adv)).cuda()
        #---------------------------------
        
        if (epoch + 1) >= args.begin_epoch:
            
            loss = nn.CrossEntropyLoss(reduce=False)(logit, target)
            # Calculate weight assignment according to geometry value
            #---------------------------------------------------
            if args.gair:
                Kappa = Kappa.cuda()
                normalized_reweight = GAIR(args.num_steps, Kappa, Lambda, args.weight_assignment_function)
#             print(normalized_reweight)
            #---------------------------------------------------
            loss = loss.mul(normalized_reweight).mean()
        else:
            loss = nn.CrossEntropyLoss(reduce="mean")(logit, target)
        
        train_robust_loss += loss.item() * len(x_adv)
        
        loss.backward()
        optimizer.step()
        
        num_data += len(data)

    train_robust_loss = train_robust_loss / num_data

    
    #-----------------------------------------
    # np.save(os.path.join(out_dir, 'delta_logits_all_' + str(epoch)), delta_logits_all.cpu().numpy())
    # np.save(os.path.join(out_dir, 'correct_clean_all_' + str(epoch)), correct_clean_all.cpu().numpy())
    # np.save(os.path.join(out_dir, 'correct_early_all_' + str(epoch)), correct_early_all.cpu().numpy())
    # np.save(os.path.join(out_dir, 'label_true_' + str(epoch)), label_true.cpu().numpy())
    #-----------------------------------------
    
    return train_robust_loss, lr

# Adjust lambda for weight assignment using epoch
def adjust_Lambda(epoch):
    Lam = float(args.Lambda)
    if args.epochs >= 110:
        # Train Wide-ResNet
        Lambda = args.Lambda_max
        if args.Lambda_schedule == 'linear':
            if epoch >= 60:
                Lambda = args.Lambda_max - (epoch/args.epochs) * (args.Lambda_max - Lam)
        elif args.Lambda_schedule == 'piecewise':
            if epoch >= 60:
                Lambda = Lam
            elif epoch >= 90:
                Lambda = Lam-1.0
            elif epoch >= 110:
                Lambda = Lam-1.5
        elif args.Lambda_schedule == 'fixed':
            if epoch >= 60:
                Lambda = Lam
    else:
        # Train ResNet
        Lambda = args.Lambda_max
        if args.Lambda_schedule == 'linear':
            if epoch >= 30:
                Lambda = args.Lambda_max - (epoch/args.epochs) * (args.Lambda_max - Lam)
        elif args.Lambda_schedule == 'piecewise':
            if epoch >= 30:
                Lambda = Lam
            elif epoch >= 60:
                Lambda = Lam-2.0
        elif args.Lambda_schedule == 'fixed':
            if epoch >= 30:
                Lambda = Lam
    return Lambda

# Setup data loader
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

if args.dataset == "cifar10":
    trainset = torchvision.datasets.CIFAR10(root='./data/cifar-10', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data/cifar-10', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
if args.dataset == "svhn":
    trainset = torchvision.datasets.SVHN(root='./data/SVHN', split='train', download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testset = torchvision.datasets.SVHN(root='./data/SVHN', split='test', download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
if args.dataset == "mnist":
    trainset = torchvision.datasets.MNIST(root='./data/MNIST', train=True, download=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=1,pin_memory=True)
    testset = torchvision.datasets.MNIST(root='./data/MNIST', train=False, download=True, transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=1,pin_memory=True)

# Resume 
title = 'GAIRAT'
best_acc = 0
start_epoch = 0
if resume:
    # Resume directly point to checkpoint.pth.tar
    print ('==> GAIRAT Resuming from checkpoint ..')
    print(resume)
    assert os.path.isfile(resume)
    #================================
    # 4.26 16:15
    #out_dir = os.path.dirname(resume)
    shutil.copy(os.path.join(os.path.dirname(resume), 'log_results.txt'), out_dir)
    #================================
    checkpoint = torch.load(resume)
    start_epoch = checkpoint['epoch']
    best_acc = checkpoint['test_pgd20_acc']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    logger_test = Logger(os.path.join(out_dir, 'log_results.txt'), title=title, resume=True)
else:
    print('==> GAIRAT')
    logger_test = Logger(os.path.join(out_dir, 'log_results.txt'), title=title)
    logger_test.set_names(['Epoch', 'Natural Test Acc', 'PGD20 Acc'])

## Training get started
test_nat_acc = 0
test_pgd20_acc = 0

for epoch in range(start_epoch, args.epochs):
    
    # Get lambda
    Lambda = adjust_Lambda(epoch + 1)
    
    # Adversarial training
    time_start = time.time()
    train_robust_loss, lr = train(epoch, model, train_loader, optimizer, Lambda)
    time_end = time.time()
    with open(file = os.path.join(out_dir, 'timer.txt'), mode = "a+", encoding = "utf-8") as f:
        f.write(str(epoch) + " " + str(round(time_end - time_start)) + '\n')
    # Evalutions similar to DAT.
    #-------------------------------------------------------------
    # 5.9 20:51
    # _, test_nat_acc = attack.eval_clean(model, test_loader)
    # _, test_pgd20_acc = attack.eval_robust(model, test_loader, perturb_steps=20, epsilon=0.031, step_size=0.031 / 4,loss_fn="cent", category="Madry", random=True)
    _, test_nat_acc, test_clean_logits_all, test_label_true = attack.eval_clean_tj(model, test_loader)
    _, test_pgd20_acc, test_adv_logits_all = attack.eval_robust_tj(model, test_loader, perturb_steps=20, epsilon=0.031, step_size=0.031 / 4,loss_fn="cent", category="Madry", random=True)
    #-----------------------------------------
    # np.save(os.path.join(out_dir, 'test_clean_logits_all_' + str(epoch)), test_clean_logits_all.cpu().numpy())
    # np.save(os.path.join(out_dir, 'test_adv_logits_all_' + str(epoch)), test_adv_logits_all.cpu().numpy())
    # if epoch == start_epoch:
    #     np.save(os.path.join(out_dir, 'test_label_true'), test_label_true.cpu().numpy())
    #-----------------------------------------
    #-------------------------------------------------------------


    print(
        'Epoch: [%d | %d] | Learning Rate: %f | Natural Test Acc %.2f | PGD20 Test Acc %.2f |\n' % (
        epoch,
        args.epochs,
        lr,
        test_nat_acc,
        test_pgd20_acc)
        )
         
    logger_test.append([epoch + 1, test_nat_acc, test_pgd20_acc])
    
    # Save the best checkpoint
    if test_pgd20_acc > best_acc:
        best_acc = test_pgd20_acc
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'test_nat_acc': test_nat_acc, 
                'test_pgd20_acc': test_pgd20_acc,
                'optimizer' : optimizer.state_dict(),
            },filename='bestpoint.pth.tar')

    # Save the last checkpoint
    save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'test_nat_acc': test_nat_acc, 
                'test_pgd20_acc': test_pgd20_acc,
                'optimizer' : optimizer.state_dict(),
            #=================
            #})
            },filename=str(epoch + 1)+'.pth.tar')
            #=================
    
    
    
logger_test.close()