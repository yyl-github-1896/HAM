import numpy as np
from models import *
from torch.autograd import Variable

#======================================
# 2.27 16:31
cifar100_mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343) # equals np.mean(train_set.train_data, axis=(0,1,2))/255
cifar100_std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404) # equals np.std(train_set.train_data, axis=(0,1,2))/255
mu_cifar100 = torch.tensor(cifar100_mean).view(3,1,1).cuda()
std_cifar100 = torch.tensor(cifar100_std).view(3,1,1).cuda()
def normalize(X):
    return (X - mu_cifar100)/std_cifar100
    # return X
#======================================

def cwloss(output, target,confidence=50, num_classes=10):
    # Compute the probability of the label class versus the maximum other
    # The same implementation as in repo CAT https://github.com/sunblaze-ucb/curriculum-adversarial-training-CAT
    target = target.data
    target_onehot = torch.zeros(target.size() + (num_classes,))
    target_onehot = target_onehot.cuda()
    target_onehot.scatter_(1, target.unsqueeze(1), 1.)
    target_var = Variable(target_onehot, requires_grad=False)
    real = (target_var * output).sum(1)
    other = ((1. - target_var) * output - target_var * 10000.).max(1)[0]
    loss = -torch.clamp(real - other + confidence, min=0.)  # equiv to max(..., 0.)
    loss = torch.sum(loss)
    return loss

def GA_PGD_dlr(model, data, target, epsilon, step_size, num_steps,loss_fn,category,rand_init):
    model.eval()
    Kappa = torch.zeros(len(data))
    if category == "trades":
        x_adv = data.detach() + 0.001 * torch.randn(data.shape).cuda().detach() if rand_init else data.detach()
        #--------------------------
        # 4.17 22:00
#         nat_output = model(data)
        nat_output = model(normalize(data))
        #--------------------------
    if category == "Madry":
        x_adv = data.detach() + torch.from_numpy(np.random.uniform(-epsilon, epsilon, data.shape)).float().cuda() if rand_init else data.detach()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    for k in range(num_steps):
        x_adv.requires_grad_()
        #--------------------------
        # 4.17 22:00
#         output = model(x_adv)
        output = model(normalize(x_adv))
        #--------------------------
        predict = output.max(1, keepdim=True)[1]
        # Update Kappa
        for p in range(len(x_adv)):
            if predict[p] == target[p]:
                Kappa[p] += 1
        model.zero_grad()
        with torch.enable_grad():
            if loss_fn == "cent":
                loss_adv = nn.CrossEntropyLoss(reduction="mean")(output, target)
            if loss_fn == "cw":
                loss_adv = cwloss(output,target)
            if loss_fn == "kl":
                criterion_kl = nn.KLDivLoss(size_average=False).cuda()
                loss_adv = criterion_kl(F.log_softmax(output, dim=1),F.softmax(nat_output, dim=1))
            #------------------------------------------------
            # 10.25 22
            if loss_fn == "dlr":
                loss_adv = dlr_loss(output, target).mean()
            #------------------------------------------------
        loss_adv.backward()
        eta = step_size * x_adv.grad.sign()
        # Update adversarial data
        x_adv = x_adv.detach() + eta
        x_adv = torch.min(torch.max(x_adv, data - epsilon), data + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    x_adv = Variable(x_adv, requires_grad=False)
    return x_adv, Kappa

# Geometry-aware projected gradient descent (GA-PGD)
def GA_PGD(model, data, target, epsilon, step_size, num_steps,loss_fn,category,rand_init):
    model.eval()
    Kappa = torch.zeros(len(data))
    if category == "trades":
        x_adv = data.detach() + 0.001 * torch.randn(data.shape).cuda().detach() if rand_init else data.detach()
        #--------------------------
        # 4.17 22:00
#         nat_output = model(data)
        nat_output = model(normalize(data))
        #--------------------------
    if category == "Madry":
        x_adv = data.detach() + torch.from_numpy(np.random.uniform(-epsilon, epsilon, data.shape)).float().cuda() if rand_init else data.detach()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    for k in range(num_steps):
        x_adv.requires_grad_()
        #--------------------------
        # 4.17 22:00
#         output = model(x_adv)
        output = model(normalize(x_adv))
        #--------------------------
        predict = output.max(1, keepdim=True)[1]
        # Update Kappa
        for p in range(len(x_adv)):
            if predict[p] == target[p]:
                Kappa[p] += 1
        model.zero_grad()
        with torch.enable_grad():
            if loss_fn == "cent":
                loss_adv = nn.CrossEntropyLoss(reduction="mean")(output, target)
            if loss_fn == "cw":
                loss_adv = cwloss(output,target)
            if loss_fn == "kl":
                criterion_kl = nn.KLDivLoss(size_average=False).cuda()
                loss_adv = criterion_kl(F.log_softmax(output, dim=1),F.softmax(nat_output, dim=1))
        loss_adv.backward()
        eta = step_size * x_adv.grad.sign()
        # Update adversarial data
        x_adv = x_adv.detach() + eta
        x_adv = torch.min(torch.max(x_adv, data - epsilon), data + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    x_adv = Variable(x_adv, requires_grad=False)
    return x_adv, Kappa


    # return x_adv, Kappa, w_logits
#-------------------------------
# 10.25 20
def dlr_loss(x, y):
    x_sorted, ind_sorted = x.sort(dim=1)
    ind = (ind_sorted[:, -1] == y).float()
    u = torch.arange(x.shape[0])
    return -(x[u, y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (
        1. - ind)) / (x_sorted[:, -1] - x_sorted[:, -3] + 1e-12)
#-------------------------------
#======================================
def GA_PGD_early2_dlr(model, data, target, epsilon, step_size, num_steps,loss_fn,category,rand_init,classify_step):
    model.eval()
    ind = 0
    if category == "trades":
        x_adv = data.detach() + 0.001 * torch.randn(data.shape).cuda().detach() if rand_init else data.detach()
        #--------------------------
        # 4.17 22:00
#         nat_output = model(data)
        nat_output = model(normalize(data))
        #--------------------------
    if category == "Madry":
        x_adv = data.detach() + torch.from_numpy(np.random.uniform(-epsilon, epsilon, data.shape)).float().cuda() if rand_init else data.detach()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    softmax = nn.Softmax(dim=1)
    logits_clean = model(normalize(data)).detach()
    logits_adv = model(normalize(x_adv)).detach()
    delta_max = torch.norm(softmax(logits_clean) - softmax(logits_adv), p=1, dim=1).detach()
    for k in range(num_steps):
        x_adv.requires_grad_()
        #--------------------------
        # 4.17 22:00
#         output = model(x_adv)
        # 10.25 20
        output = model(normalize(x_adv))
        # output = model(normalize(x_adv)) * large_logits
        #--------------------------
        predict = output.max(1, keepdim=True)[1]
        model.zero_grad()
        with torch.enable_grad():
            if loss_fn == "cent":
                loss_adv = nn.CrossEntropyLoss(reduction="mean")(output, target)
            if loss_fn == "cw":
                loss_adv = cwloss(output,target)
            if loss_fn == "kl":
                criterion_kl = nn.KLDivLoss(size_average=False).cuda()
                loss_adv = criterion_kl(F.log_softmax(output, dim=1),F.softmax(nat_output, dim=1))
            #------------------------------------------------
            # 10.25 22
            if loss_fn == "dlr":
                loss_adv = dlr_loss(output, target).mean()
            #------------------------------------------------
        loss_adv.backward()
        eta = step_size * x_adv.grad.sign()
        # Update adversarial data
        x_adv = x_adv.detach() + eta
        x_adv = torch.min(torch.max(x_adv, data - epsilon), data + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
        ind += 1
        logits_adv = model(normalize(x_adv)).detach()
        if ind == classify_step:
            
            pred_adv = logits_adv.max(1, keepdim=True)[1]
            correct_early = pred_adv.eq(target.view_as(pred_adv)).float().squeeze(dim=1)
            
            w_logits = torch.zeros(len(target))
            w_logits[torch.where(correct_early == 0)] = 1
            if w_logits.sum() == 0:
                return x_adv, w_logits
            x_adv = x_adv[torch.where(correct_early == 0)].detach()
            target = target[torch.where(correct_early == 0)].detach()
            data = data[torch.where(correct_early == 0)].detach()
    x_adv = Variable(x_adv, requires_grad=False)
    # w_logits代表的是哪些需要保留，哪些需要丢弃，x_adv是对抗样本
    return x_adv, w_logits

def GA_PGD_early2_large_logits(model, data, target, epsilon, step_size, num_steps,loss_fn,category,rand_init,classify_step, large_logits):
    model.eval()
    ind = 0
    if category == "trades":
        x_adv = data.detach() + 0.001 * torch.randn(data.shape).cuda().detach() if rand_init else data.detach()
        #--------------------------
        # 4.17 22:00
#         nat_output = model(data)
        nat_output = model(normalize(data))
        #--------------------------
    if category == "Madry":
        x_adv = data.detach() + torch.from_numpy(np.random.uniform(-epsilon, epsilon, data.shape)).float().cuda() if rand_init else data.detach()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    softmax = nn.Softmax(dim=1)
    logits_clean = model(normalize(data)).detach()
    logits_adv = model(normalize(x_adv)).detach()
    delta_max = torch.norm(softmax(logits_clean) - softmax(logits_adv), p=1, dim=1).detach()
    for k in range(num_steps):
        x_adv.requires_grad_()
        #--------------------------
        # 4.17 22:00
#         output = model(x_adv)
        # 10.25 20
        # output = model(normalize(x_adv))
        output = model(normalize(x_adv)) * large_logits
        #--------------------------
        predict = output.max(1, keepdim=True)[1]
        model.zero_grad()
        with torch.enable_grad():
            if loss_fn == "cent":
                loss_adv = nn.CrossEntropyLoss(reduction="mean")(output, target)
            if loss_fn == "cw":
                loss_adv = cwloss(output,target)
            if loss_fn == "kl":
                criterion_kl = nn.KLDivLoss(size_average=False).cuda()
                loss_adv = criterion_kl(F.log_softmax(output, dim=1),F.softmax(nat_output, dim=1))
        loss_adv.backward()
        eta = step_size * x_adv.grad.sign()
        # Update adversarial data
        x_adv = x_adv.detach() + eta
        x_adv = torch.min(torch.max(x_adv, data - epsilon), data + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
        ind += 1
        logits_adv = model(normalize(x_adv)).detach()
        if ind == classify_step:
            
            pred_adv = logits_adv.max(1, keepdim=True)[1]
            correct_early = pred_adv.eq(target.view_as(pred_adv)).float().squeeze(dim=1)
            
            w_logits = torch.zeros(len(target))
            w_logits[torch.where(correct_early == 0)] = 1
            if w_logits.sum() == 0:
                return x_adv, w_logits
            x_adv = x_adv[torch.where(correct_early == 0)].detach()
            target = target[torch.where(correct_early == 0)].detach()
            data = data[torch.where(correct_early == 0)].detach()
    x_adv = Variable(x_adv, requires_grad=False)
    return x_adv, w_logits

def GA_PGD_early2(model, data, target, epsilon, step_size, num_steps,loss_fn,category,rand_init,classify_step):
    model.eval()
    Kappa = torch.zeros(len(data))
    
    #----------------------------
#     delta_logits = torch.zeros(len(data), num_steps + 1, 10).cuda()
#     delta_p = torch.zeros(len(data), num_steps + 1, 10).cuda()
    ind = 0
    
#     softmax = nn.Softmax(dim=1)
#     logits_adv_last = model(normalize(data)).detach()
#     delta_logits[:,ind, :] = logits_adv_last
    
#     pred = logits_adv_last.max(1, keepdim=True)[1]
#     correct_clean = pred.eq(target.view_as(pred)).float().squeeze(dim=1)
    #----------------------------
    
    if category == "trades":
        x_adv = data.detach() + 0.001 * torch.randn(data.shape).cuda().detach() if rand_init else data.detach()
        #--------------------------
        # 4.17 22:00
#         nat_output = model(data)
        nat_output = model(normalize(data))
        #--------------------------
    if category == "Madry":
        x_adv = data.detach() + torch.from_numpy(np.random.uniform(-epsilon, epsilon, data.shape)).float().cuda() if rand_init else data.detach()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    softmax = nn.Softmax(dim=1)
    logits_clean = model(normalize(data)).detach()
    logits_adv = model(normalize(x_adv)).detach()
    delta_max = torch.norm(softmax(logits_clean) - softmax(logits_adv), p=1, dim=1).detach()
    for k in range(num_steps):
        x_adv.requires_grad_()
        #--------------------------
        # 4.17 22:00
#         output = model(x_adv)
        output = model(normalize(x_adv))
        #--------------------------
        predict = output.max(1, keepdim=True)[1]
        # Update Kappa
        # for p in range(len(x_adv)):
        #     if predict[p] == target[p]:
        #         Kappa[p] += 1
        model.zero_grad()
        with torch.enable_grad():
            if loss_fn == "cent":
                loss_adv = nn.CrossEntropyLoss(reduction="mean")(output, target)
            if loss_fn == "cw":
                loss_adv = cwloss(output,target)
            if loss_fn == "kl":
                criterion_kl = nn.KLDivLoss(size_average=False).cuda()
                loss_adv = criterion_kl(F.log_softmax(output, dim=1),F.softmax(nat_output, dim=1))
        loss_adv.backward()
        eta = step_size * x_adv.grad.sign()
        # Update adversarial data
        x_adv = x_adv.detach() + eta
        x_adv = torch.min(torch.max(x_adv, data - epsilon), data + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
        #----------------------------
        # 8.24
        # output_this = model(normalize(x_adv)).detach()
        # delta_this = torch.norm(softmax(output_this) - softmax(output), p=1, dim=1).detach()
        # delta_max = torch.where(delta_max > delta_this, delta_max, delta_this)
        #----------------------------
        ind += 1
        logits_adv = model(normalize(x_adv)).detach()
        # delta_logits[:,ind, :] = logits_adv
        if ind == classify_step:
            
            pred_adv = logits_adv.max(1, keepdim=True)[1]
            correct_early = pred_adv.eq(target.view_as(pred_adv)).float().squeeze(dim=1)
            
            w_logits = torch.zeros(len(target))
            w_logits[torch.where(correct_early == 0)] = 1
            if w_logits.sum() == 0:
                return x_adv, w_logits
            # w_logits为0表示早期丢弃，easyAE，为1表示hard AE
            # 如下是ablation study：分别计算Hard AE和Easy AE的梯度和cos similarity
            # x_adv = x_adv.detach()
            # target = target.detach()
            # data = data.detach()
            # 如下是原version
            x_adv = x_adv[torch.where(correct_early == 0)].detach()
            # print(x_adv.shape)
            target = target[torch.where(correct_early == 0)].detach()
            data = data[torch.where(correct_early == 0)].detach()
            # delta_max = delta_max[torch.where(correct_early == 0)].detach()
        #----------------------------
        
        
    x_adv = Variable(x_adv, requires_grad=False)
    # return x_adv, Kappa, w_logits, delta_logits, correct_clean, correct_early
    return x_adv, w_logits

def GA_PGD_early2re(model, data, target, epsilon, step_size, num_steps,loss_fn,category,rand_init,classify_step):
    model.eval()
    Kappa = torch.zeros(len(data))
    
    #----------------------------
#     delta_logits = torch.zeros(len(data), num_steps + 1, 10).cuda()
#     delta_p = torch.zeros(len(data), num_steps + 1, 10).cuda()
    ind = 0
    
#     softmax = nn.Softmax(dim=1)
#     logits_adv_last = model(normalize(data)).detach()
#     delta_logits[:,ind, :] = logits_adv_last
    
#     pred = logits_adv_last.max(1, keepdim=True)[1]
#     correct_clean = pred.eq(target.view_as(pred)).float().squeeze(dim=1)
    #----------------------------
    
    if category == "trades":
        x_adv = data.detach() + 0.001 * torch.randn(data.shape).cuda().detach() if rand_init else data.detach()
        #--------------------------
        # 4.17 22:00
#         nat_output = model(data)
        nat_output = model(normalize(data))
        #--------------------------
    if category == "Madry":
        x_adv = data.detach() + torch.from_numpy(np.random.uniform(-epsilon, epsilon, data.shape)).float().cuda() if rand_init else data.detach()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    softmax = nn.Softmax(dim=1)
    logits_clean = model(normalize(data)).detach()
    logits_adv = model(normalize(x_adv)).detach()
    delta_max = torch.norm(softmax(logits_clean) - softmax(logits_adv), p=1, dim=1).detach()
    for k in range(num_steps):
        x_adv.requires_grad_()
        #--------------------------
        # 4.17 22:00
#         output = model(x_adv)
        output = model(normalize(x_adv))
        #--------------------------
        predict = output.max(1, keepdim=True)[1]
        # Update Kappa
        # for p in range(len(x_adv)):
        #     if predict[p] == target[p]:
        #         Kappa[p] += 1
        model.zero_grad()
        with torch.enable_grad():
            if loss_fn == "cent":
                loss_adv = nn.CrossEntropyLoss(reduction="mean")(output, target)
            if loss_fn == "cw":
                loss_adv = cwloss(output,target)
            if loss_fn == "kl":
                criterion_kl = nn.KLDivLoss(size_average=False).cuda()
                loss_adv = criterion_kl(F.log_softmax(output, dim=1),F.softmax(nat_output, dim=1))
        loss_adv.backward()
        eta = step_size * x_adv.grad.sign()
        # Update adversarial data
        x_adv = x_adv.detach() + eta
        x_adv = torch.min(torch.max(x_adv, data - epsilon), data + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
        #----------------------------
        # 8.24
        # output_this = model(normalize(x_adv)).detach()
        # delta_this = torch.norm(softmax(output_this) - softmax(output), p=1, dim=1).detach()
        # delta_max = torch.where(delta_max > delta_this, delta_max, delta_this)
        #----------------------------
        ind += 1
        logits_adv = model(normalize(x_adv)).detach()
        # delta_logits[:,ind, :] = logits_adv
        if ind == classify_step:
            
            pred_adv = logits_adv.max(1, keepdim=True)[1]
            correct_early = pred_adv.eq(target.view_as(pred_adv)).float().squeeze(dim=1)
            
            w_logits = torch.zeros(len(target))
            w_logits[torch.where(correct_early == 0)] = 1
            if w_logits.sum() == 0:
                return x_adv, w_logits
            # w_logits为0表示早期丢弃，easyAE，为1表示hard AE
            # 如下是ablation study：分别计算Hard AE和Easy AE的梯度和cos similarity
            # x_adv = x_adv.detach()
            # target = target.detach()
            # data = data.detach()
            # 如下是原version
            x_adv = x_adv[torch.where(correct_early ==0)].detach()
            print(x_adv.shape)
            target = target[torch.where(correct_early == 0)].detach()
            data = data[torch.where(correct_early == 0)].detach()
            # delta_max = delta_max[torch.where(correct_early == 0)].detach()
        #----------------------------
        
        
    x_adv = Variable(x_adv, requires_grad=False)
    # return x_adv, Kappa, w_logits, delta_logits, correct_clean, correct_early
    return x_adv, w_logits

def g111(model, data, target, epsilon, step_size, num_steps,loss_fn,category,rand_init,classify_step):
    model.eval()
    Kappa = torch.zeros(len(data))
    
    #----------------------------
#     delta_logits = torch.zeros(len(data), num_steps + 1, 10).cuda()
#     delta_p = torch.zeros(len(data), num_steps + 1, 10).cuda()
    ind = 0
    
#     softmax = nn.Softmax(dim=1)
#     logits_adv_last = model(normalize(data)).detach()
#     delta_logits[:,ind, :] = logits_adv_last
    
#     pred = logits_adv_last.max(1, keepdim=True)[1]
#     correct_clean = pred.eq(target.view_as(pred)).float().squeeze(dim=1)
    #----------------------------
    
    if category == "trades":
        x_adv = data.detach() + 0.001 * torch.randn(data.shape).cuda().detach() if rand_init else data.detach()
        #--------------------------
        # 4.17 22:00
#         nat_output = model(data)
        nat_output = model(normalize(data))
        #--------------------------
    if category == "Madry":
        x_adv = data.detach() + torch.from_numpy(np.random.uniform(-epsilon, epsilon, data.shape)).float().cuda() if rand_init else data.detach()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    softmax = nn.Softmax(dim=1)
    logits_clean = model(normalize(data)).detach()
    logits_adv = model(normalize(x_adv)).detach()
    delta_max = torch.norm(softmax(logits_clean) - softmax(logits_adv), p=1, dim=1).detach()
    for k in range(num_steps):
        x_adv.requires_grad_()
        #--------------------------
        # 4.17 22:00
#         output = model(x_adv)
        output = model(normalize(x_adv))
        #--------------------------
        predict = output.max(1, keepdim=True)[1]
        # Update Kappa
        # for p in range(len(x_adv)):
        #     if predict[p] == target[p]:
        #         Kappa[p] += 1
        model.zero_grad()
        with torch.enable_grad():
            if loss_fn == "cent":
                loss_adv = nn.CrossEntropyLoss(reduction="mean")(output, target)
            if loss_fn == "cw":
                loss_adv = cwloss(output,target)
            if loss_fn == "kl":
                criterion_kl = nn.KLDivLoss(size_average=False).cuda()
                loss_adv = criterion_kl(F.log_softmax(output, dim=1),F.softmax(nat_output, dim=1))
        loss_adv.backward()
        eta = step_size * x_adv.grad.sign()
        # Update adversarial data
        x_adv = x_adv.detach() + eta
        x_adv = torch.min(torch.max(x_adv, data - epsilon), data + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
        #----------------------------
        # 8.24
        # output_this = model(normalize(x_adv)).detach()
        # delta_this = torch.norm(softmax(output_this) - softmax(output), p=1, dim=1).detach()
        # delta_max = torch.where(delta_max > delta_this, delta_max, delta_this)
        #----------------------------
        ind += 1
        logits_adv = model(normalize(x_adv)).detach()
        # delta_logits[:,ind, :] = logits_adv
        if ind == classify_step:
            
            pred_adv = logits_adv.max(1, keepdim=True)[1]
            correct_early = pred_adv.eq(target.view_as(pred_adv)).float().squeeze(dim=1)
            
            w_logits = torch.zeros(len(target))
            w_logits[torch.where(correct_early == 0)] = 1
            print(w_logits.sum())
            if w_logits.sum() == 0:
                return x_adv, w_logits
            # w_logits为0表示早期丢弃，easyAE，为1表示hard AE
            # 如下是ablation study：分别计算Hard AE和Easy AE的梯度和cos similarity
            x_adv_easy = x_adv[torch.where(correct_early == 1)].detach()
            # print(x_adv.shape)
            target_easy = target[torch.where(correct_early == 1)].detach()
            data_easy = data[torch.where(correct_early == 1)].detach()
            x_adv = x_adv[torch.where(correct_early == 0)].detach()
            print(x_adv.shape)
            print(x_adv_easy.shape)
            # print(target.shape)
            # print(data.shape)
            target = target[torch.where(correct_early == 0)].detach()
            data = data[torch.where(correct_early == 0)].detach()
            # # 如下是原version
            # x_adv = x_adv[torch.where(correct_early == 0)].detach()
            # # print(x_adv.shape)
            # target = target[torch.where(correct_early == 0)].detach()
            # data = data[torch.where(correct_early == 0)].detach()
            # # delta_max = delta_max[torch.where(correct_early == 0)].detach()
        if ind>classify_step:
            x_adv_easy.requires_grad_()
            #--------------------------
            # 4.17 22:00
    #         output = model(x_adv)
            output = model(normalize(x_adv_easy))
            #--------------------------
            predict = output.max(1, keepdim=True)[1]
            # Update Kappa
            # for p in range(len(x_adv)):
            #     if predict[p] == target[p]:
            #         Kappa[p] += 1
            model.zero_grad()
            with torch.enable_grad():
                if loss_fn == "cent":
                    loss_adv = nn.CrossEntropyLoss(reduction="mean")(output, target_easy)
                if loss_fn == "cw":
                    loss_adv = cwloss(output,target_easy)
                if loss_fn == "kl":
                    criterion_kl = nn.KLDivLoss(size_average=False).cuda()
                    loss_adv = criterion_kl(F.log_softmax(output, dim=1),F.softmax(nat_output, dim=1))
            loss_adv.backward()
            eta = step_size * x_adv_easy.grad.sign()
            # Update adversarial data
            x_adv_easy = x_adv_easy.detach() + eta
            x_adv_easy = torch.min(torch.max(x_adv_easy, data_easy - epsilon), data_easy + epsilon)
            x_adv_easy = torch.clamp(x_adv_easy, 0.0, 1.0)
        
        
    x_adv = Variable(x_adv, requires_grad=False)
    x_adv_easy= Variable(x_adv_easy, requires_grad=False)
    # return x_adv, Kappa, w_logits, delta_logits, correct_clean, correct_early
    return x_adv, w_logits,x_adv_easy,target,target_easy

def GA_PGD_early(model, data, target, epsilon, step_size, num_steps,loss_fn,category,rand_init,classify_step):
    model.eval()
    Kappa = torch.zeros(len(data))
    
    #----------------------------
    delta_logits = torch.zeros(len(data), num_steps + 1, 10).cuda()
    delta_p = torch.zeros(len(data), num_steps + 1, 10).cuda()
    ind = 0
    
    softmax = nn.Softmax(dim=1)
    logits_adv_last = model(normalize(data)).detach()
    delta_logits[:,ind, :] = logits_adv_last
    
    pred = logits_adv_last.max(1, keepdim=True)[1]
    correct_clean = pred.eq(target.view_as(pred)).float().squeeze(dim=1)
    #----------------------------
    
    if category == "trades":
        x_adv = data.detach() + 0.001 * torch.randn(data.shape).cuda().detach() if rand_init else data.detach()
        #--------------------------
        # 4.17 22:00
#         nat_output = model(data)
        nat_output = model(normalize(data))
        #--------------------------
    if category == "Madry":
        x_adv = data.detach() + torch.from_numpy(np.random.uniform(-epsilon, epsilon, data.shape)).float().cuda() if rand_init else data.detach()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    for k in range(num_steps):
        x_adv.requires_grad_()
        #--------------------------
        # 4.17 22:00
#         output = model(x_adv)
        output = model(normalize(x_adv))
        #--------------------------
        predict = output.max(1, keepdim=True)[1]
        # Update Kappa
        for p in range(len(x_adv)):
            if predict[p] == target[p]:
                Kappa[p] += 1
        model.zero_grad()
        with torch.enable_grad():
            if loss_fn == "cent":
                loss_adv = nn.CrossEntropyLoss(reduction="mean")(output, target)
            if loss_fn == "cw":
                loss_adv = cwloss(output,target)
            if loss_fn == "kl":
                criterion_kl = nn.KLDivLoss(size_average=False).cuda()
                loss_adv = criterion_kl(F.log_softmax(output, dim=1),F.softmax(nat_output, dim=1))
        loss_adv.backward()
        eta = step_size * x_adv.grad.sign()
        # Update adversarial data
        x_adv = x_adv.detach() + eta
        x_adv = torch.min(torch.max(x_adv, data - epsilon), data + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
        #----------------------------
        ind += 1
        logits_adv = model(normalize(x_adv)).detach()
        delta_logits[:,ind, :] = logits_adv
        if ind == classify_step:
            
            pred_adv = logits_adv.max(1, keepdim=True)[1]
            correct_early = pred_adv.eq(target.view_as(pred_adv)).float().squeeze(dim=1)
            w_logits = torch.zeros(len(target))
            w_logits[torch.where(correct_early == 0)] = 1
        #----------------------------
        
        
    x_adv = Variable(x_adv, requires_grad=False)
    return x_adv, Kappa, w_logits, delta_logits, correct_clean, correct_early
    
    
def GA_PGD_tj(model, data, target, epsilon, step_size, num_steps,loss_fn,category,rand_init):
    model.eval()
    Kappa = torch.zeros(len(data))
    
    #----------------------------
    delta_logits = torch.zeros(len(data), num_steps + 1, 10).cuda()
    delta_p = torch.zeros(len(data), num_steps + 1, 10).cuda()
    ind = 0
    
    softmax = nn.Softmax(dim=1)
    logits_adv_last = model(normalize(data)).detach()
    delta_logits[:,ind, :] = logits_adv_last
    delta_p[:,ind, :] = softmax(logits_adv_last).detach()
    
    pred = logits_adv_last.max(1, keepdim=True)[1]
    correct_clean = pred.eq(target.view_as(pred)).float().squeeze(dim=1)
    #----------------------------
    
    if category == "trades":
        x_adv = data.detach() + 0.001 * torch.randn(data.shape).cuda().detach() if rand_init else data.detach()
        #--------------------------
        # 4.17 22:00
#         nat_output = model(data)
        nat_output = model(normalize(data))
        #--------------------------
    if category == "Madry":
        x_adv = data.detach() + torch.from_numpy(np.random.uniform(-epsilon, epsilon, data.shape)).float().cuda() if rand_init else data.detach()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    for k in range(num_steps):
        x_adv.requires_grad_()
        #--------------------------
        # 4.17 22:00
#         output = model(x_adv)
        output = model(normalize(x_adv))
        #--------------------------
        predict = output.max(1, keepdim=True)[1]
        # Update Kappa
        for p in range(len(x_adv)):
            if predict[p] == target[p]:
                Kappa[p] += 1
        model.zero_grad()
        with torch.enable_grad():
            if loss_fn == "cent":
                loss_adv = nn.CrossEntropyLoss(reduction="mean")(output, target)
            if loss_fn == "cw":
                loss_adv = cwloss(output,target)
            if loss_fn == "kl":
                criterion_kl = nn.KLDivLoss(size_average=False).cuda()
                loss_adv = criterion_kl(F.log_softmax(output, dim=1),F.softmax(nat_output, dim=1))
        loss_adv.backward()
        eta = step_size * x_adv.grad.sign()
        # Update adversarial data
        x_adv = x_adv.detach() + eta
        x_adv = torch.min(torch.max(x_adv, data - epsilon), data + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
        #----------------------------
        ind += 1
        logits_adv_this = model(normalize(x_adv)).detach()
        #----------------------------------------------------------------------
        # 7.18 23:02
        # delta_logits[:,ind, :] = (logits_adv_this - logits_adv_last).detach()
        delta_logits[:,ind, :] = logits_adv_this.detach()
        #----------------------------------------------------------------------
        delta_p[:,ind, :] = (softmax(logits_adv_this) - softmax(logits_adv_last)).detach()
        
        logits_adv_last = logits_adv_this
        #----------------------------
        
        
    x_adv = Variable(x_adv, requires_grad=False)
    #----------------------------
    pred_adv = model(normalize(x_adv)).max(1, keepdim=True)[1].detach()
    correct_adv = pred_adv.eq(target.view_as(pred_adv)).float().squeeze(dim=1)
    #----------------------------
    return x_adv, Kappa, delta_logits, delta_p, correct_clean, correct_adv

def GA_PGD_delta(model, data, target, epsilon, step_size, num_steps,loss_fn,category,rand_init,way, classify_step):
    model.eval()
    Kappa = torch.zeros(len(data))

    #----------------------
    # 8.06
    ind = 0
    #----------------------


    if category == "trades":
        x_adv = data.detach() + 0.001 * torch.randn(data.shape).cuda().detach() if rand_init else data.detach()
        #--------------------------
        # 4.17 22:00
#         nat_output = model(data)
        nat_output = model(normalize(data))
        #--------------------------
    if category == "Madry":
        x_adv = data.detach() + torch.from_numpy(np.random.uniform(-epsilon, epsilon, data.shape)).float().cuda() if rand_init else data.detach()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    
    # ===========================================
    # 4.25 21:20
    if way == 'softmax':
        softmax = nn.Softmax(dim=1)
    else:
        def softmax(x):
            return x
    logits_clean = model(normalize(data)).detach()
    logits_adv = model(normalize(x_adv)).detach()
    delta_max = torch.norm(softmax(logits_clean) - softmax(logits_adv), p=1, dim=1).detach()
    # ===========================================
    
    for k in range(num_steps):
        x_adv.requires_grad_()
        #--------------------------
        # 4.17 22:00
#         output = model(x_adv)
        output = model(normalize(x_adv))
        #--------------------------
        predict = output.max(1, keepdim=True)[1]
        # Update Kappa
        for p in range(len(x_adv)):
            if predict[p] == target[p]:
                Kappa[p] += 1
        model.zero_grad()
        with torch.enable_grad():
            if loss_fn == "cent":
                loss_adv = nn.CrossEntropyLoss(reduction="mean")(output, target)
            if loss_fn == "cw":
                loss_adv = cwloss(output,target)
            if loss_fn == "kl":
                criterion_kl = nn.KLDivLoss(size_average=False).cuda()
                loss_adv = criterion_kl(F.log_softmax(output, dim=1),F.softmax(nat_output, dim=1))
        loss_adv.backward()
        eta = step_size * x_adv.grad.sign()
        # Update adversarial data
        x_adv = x_adv.detach() + eta
        x_adv = torch.min(torch.max(x_adv, data - epsilon), data + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
        #--------------------------
        # 4.25 21:20
        output_this = model(normalize(x_adv)).detach()
        delta_this = torch.norm(softmax(output_this) - softmax(output), p=1, dim=1).detach()
        delta_max = torch.where(delta_max > delta_this, delta_max, delta_this)
        #--------------------------
        #----------------------------
        # 8.06 21:41 
        ind += 1
        logits_adv = model(normalize(x_adv)).detach()
        # delta_logits[:,ind, :] = logits_adv
        if ind == classify_step:
            
            pred_adv = logits_adv.max(1, keepdim=True)[1]
            correct_early = pred_adv.eq(target.view_as(pred_adv)).float().squeeze(dim=1)
            
            w_logits = torch.ones(len(target))
            w_logits[torch.where(correct_early == 1)] = 0
            if w_logits.sum() == 0:
                return x_adv, Kappa, delta_max, w_logits
            x_adv = x_adv[torch.where(correct_early == 0)].detach()
            target = target[torch.where(correct_early == 0)].detach()
            data = data[torch.where(correct_early == 0)].detach()
            delta_max = delta_max[torch.where(correct_early == 0)].detach()
        #----------------------------
    x_adv = Variable(x_adv, requires_grad=False)
    return x_adv, Kappa, delta_max, w_logits

# Geometry-aware projected gradient descent (GA-PGD)
def GA_PGD_margin(model, data, target, epsilon, step_size, num_steps,loss_fn,category,rand_init, classify_step):
    model.eval()
    Kappa = torch.zeros(len(data))

    #----------------------
    # 8.06
    ind = 0
    #----------------------

    if category == "trades":
        x_adv = data.detach() + 0.001 * torch.randn(data.shape).cuda().detach() if rand_init else data.detach()
        #--------------------------
        # 4.17 22:00
#         nat_output = model(data)
        nat_output = model(normalize(data))
        #--------------------------
    if category == "Madry":
        x_adv = data.detach() + torch.from_numpy(np.random.uniform(-epsilon, epsilon, data.shape)).float().cuda() if rand_init else data.detach()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    # ===========================================
    # 4.16 22:42
    #--------------------------
    # 4.17 22:00
#     logits_nat = model(data).detach().clone()
    logits_nat = model(normalize(data)).detach().clone()
    #--------------------------
    logits_margin = torch.zeros_like(logits_nat)
    # ===========================================
    for k in range(num_steps):
        x_adv.requires_grad_()
        #--------------------------
        # 4.17 22:00
#         output = model(x_adv)
        output = model(normalize(x_adv))
        #--------------------------
        predict = output.max(1, keepdim=True)[1]
        # Update Kappa
        for p in range(len(x_adv)):
            if predict[p] == target[p]:
                Kappa[p] += 1
        model.zero_grad()
        with torch.enable_grad():
            if loss_fn == "cent":
                loss_adv = nn.CrossEntropyLoss(reduction="mean")(output, target)
            if loss_fn == "cw":
                loss_adv = cwloss(output,target)
            if loss_fn == "kl":
                criterion_kl = nn.KLDivLoss(size_average=False).cuda()
                loss_adv = criterion_kl(F.log_softmax(output, dim=1),F.softmax(nat_output, dim=1))
        loss_adv.backward()
        eta = step_size * x_adv.grad.sign()
        # Update adversarial data
        x_adv = x_adv.detach() + eta
        x_adv = torch.min(torch.max(x_adv, data - epsilon), data + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
        
        # ===========================================
        # 11.5
        #====
        # 4.12 23：20
        # logits_robust = model(normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit)))
        #--------------------------
        # 4.17 22:00
#         logits_robust = model(x_adv).detach().clone()
        logits_robust = model(normalize(x_adv)).detach().clone()
        #--------------------------
        #====
        pred_pgd = logits_robust.max(1, keepdim=True)[1]
        wrong_pgd = 1 - pred_pgd.eq(target.view_as(pred_pgd)).float().squeeze(dim=1)

        condition = ( logits_margin.sum(dim=1) == 0 ) & (wrong_pgd == 1)
        inds = torch.where(condition)
        logits_margin[inds[0], :] = logits_robust[inds[0], :]

        if k == num_steps - 1:
            condition_not = (logits_margin.sum(dim=1) == 0)
            inds_not = torch.where(condition_not)
            logits_margin[inds_not[0], :] = logits_robust[inds_not[0], :]
            
        # ===========================================
        #----------------------------
        # 8.06 21:41 
        ind += 1
        logits_adv = model(normalize(x_adv)).detach()
        # delta_logits[:,ind, :] = logits_adv
        if ind == classify_step:
            
            pred_adv = logits_adv.max(1, keepdim=True)[1]
            correct_early = pred_adv.eq(target.view_as(pred_adv)).float().squeeze(dim=1)
            
            w_logits = torch.ones(len(target))
            w_logits[torch.where(correct_early == 1)] = 0
            if w_logits.sum() == 0:
                return x_adv, Kappa, logits_margin, w_logits
            x_adv = x_adv[torch.where(correct_early == 0)].detach()
            target = target[torch.where(correct_early == 0)].detach()
            data = data[torch.where(correct_early == 0)].detach()
            logits_margin = logits_margin[torch.where(correct_early == 0)].detach()
        #----------------------------
    x_adv = Variable(x_adv, requires_grad=False)
    return x_adv, Kappa, logits_margin, w_logits

def eval_clean_tj(model, test_loader, args):
    model.eval()
    test_loss = 0
    correct = 0
    
    #==============================
    # 5.9 20:29
    if test_loader.__len__() == 204:
        len_testset = 26032
    else:
        len_testset = 10000
    if args.dataset == "cifar100":
        test_clean_logits_all = torch.zeros(len_testset, 100)
    else:
        test_clean_logits_all = torch.zeros(len_testset, 10)
    test_label_true = torch.zeros(len_testset)
    #==============================
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()
            #--------------------------
            # 4.17 22:00
#             output = model(data)
            output = model(normalize(data))
                        
            # print('data.shape')
            # print(data.shape)
            # print('output.shape')
            # print(output.shape)
            # print('test_clean_logits_all[batch_idx * 128 : batch_idx * 128 + len(data)].shape')
            # print(test_clean_logits_all[batch_idx * 128 : batch_idx * 128 + len(data)].shape)
            # print('output.detach().shape')
            # print(output.detach().shape)
            # print('batch_idx * 128')
            # print(batch_idx * 128)
            # print(batch_idx * 128 + len(data))
            
            #==============================
            test_clean_logits_all[batch_idx * 128 : batch_idx * 128 + len(data)] = output.detach()
            test_label_true[batch_idx * 128 : batch_idx * 128 + len(data)] = target
            #==============================
            #--------------------------
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy, test_clean_logits_all, test_label_true

def eval_robust_tj(model, test_loader, args, perturb_steps, epsilon, step_size, loss_fn, category, random):
    model.eval()
    test_loss = 0
    correct = 0
    
   #==============================
    # 5.9 20:29
    if args.dataset == "cifar100":
        test_adv_logits_all = torch.zeros(10000, 100)
    else:
        test_adv_logits_all = torch.zeros(10000, 10)
    #==============================
    
    with torch.enable_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()
            x_adv, _ = GA_PGD(model,data,target,epsilon,step_size,perturb_steps,loss_fn,category,rand_init=random)
            #--------------------------
            # 4.17 22:00
#             output = model(x_adv)
            output = model(normalize(x_adv))
            #==============================
            test_adv_logits_all[batch_idx * 128 : batch_idx * 128 + len(data)] = output.detach()
            #==============================
            #--------------------------
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy, test_adv_logits_all

def eval_clean(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            #--------------------------
            # 4.17 22:00
#             output = model(data)
            output = model(normalize(data))
            #--------------------------
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy

def eval_robust(model, test_loader, perturb_steps, epsilon, step_size, loss_fn, category, random):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.enable_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            x_adv, _ = GA_PGD(model,data,target,epsilon,step_size,perturb_steps,loss_fn,category,rand_init=random)
            #--------------------------
            # 4.17 22:00
#             output = model(x_adv)
            output = model(normalize(x_adv))
            #--------------------------
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy

