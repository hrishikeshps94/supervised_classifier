import argparse
import os

import pandas as pd
import torch
import torch.optim as optim
from thop import profile, clever_format
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.datasets import ImageFolder
import utils
from model import Model,Simclr_ResNet
from utils import CustomDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import mean_and_std
global best_acc
# train for one epoch to learn unique features
def train(net, data_loader, train_optimizer,scheduler):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for pos_1, pos_2, target in train_bar:
        curr_batchsize = target.shape[0]
        pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
        feature_1, out_1 = net(pos_1)
        feature_2, out_2 = net(pos_2)
        # [2*B, D]
        out = torch.cat([out_1, out_2], dim=0)
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * curr_batchsize, device=sim_matrix.device)).bool()
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * curr_batchsize, -1)

        # compute loss
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()
        scheduler.step()

        total_num += curr_batchsize
        total_loss += loss.item() * curr_batchsize
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}, lr {:.4f}'.format(epoch, epochs, total_loss / total_num,\
        optimizer.param_groups[0]['lr']))

    return total_loss / total_num


# test for one epoch, use weighted knn to find the most similar images' label to assign the test image
def test(net, memory_data_loader, test_data_loader):
    global best_acc
    net.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for data, _, target in tqdm(memory_data_loader, desc='Feature extracting'):
            feature, out = net(data.cuda(non_blocking=True))
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data, _, target in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature, out = net(data)

            total_num += data.size(0)
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = torch.mm(feature, feature_bank)
            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
            # [B, K]
            sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
            sim_weight = (sim_weight / temperature).exp()

            # counts for each class
            one_hot_label = torch.zeros(data.size(0) * k, c, device=sim_labels.device)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
            # weighted score ---> [B, C]
            pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, c) * sim_weight.unsqueeze(dim=-1), dim=1)

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((pred_labels[:, :3] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@best:{:.2f}%'
                                     .format(epoch, epochs, total_top1 / total_num * 100,best_acc))

    return total_top1 / total_num * 100, total_top5 / total_num * 100

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--feature_dim', default=512, type=int, help='Feature dim for latent vector')
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
    parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
    parser.add_argument('--batch_size', default=128, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=500, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--train_path', default='/mnt/c/Users/Hrishikesh/Desktop/hrishi/WORK/RESEARCH/2022/MCCAI-2022/DRAC/ds/unsupervised_ds/', type=str, help='Path to dataset')
    parser.add_argument('--val_path', default='/mnt/c/Users/Hrishikesh/Desktop/hrishi/WORK/RESEARCH/2022/MCCAI-2022/DRAC/ds/grading/train/', type=str, help='Path to dataset')
    parser.add_argument('--test_path', default='/mnt/c/Users/Hrishikesh/Desktop/hrishi/WORK/RESEARCH/2022/MCCAI-2022/DRAC/ds/grading/val/', type=str, help='Path to dataset')
    parser.add_argument('--root_path', default='/mnt/c/Users/Hrishikesh/Desktop/hrishi/WORK/RESEARCH/2022/MCCAI-2022/DRAC/ds/extra/', type=str, help='Path to dataset root')
    parser.add_argument('--model_type', default='resnet18', type=str, help='Model type')
    # args parse
    args = parser.parse_args()
    feature_dim, temperature, k = args.feature_dim, args.temperature, args.k
    batch_size, epochs = args.batch_size, args.epochs
    train_num_list = []
    for dir_path, _, file_names in os.walk(args.root_path):
        for f_paths in sorted(file_names):
            train_num_list.append(os.path.join(dir_path,f_paths))
    mean,std = mean_and_std(train_num_list)
    print(mean,std)
    # data prepare
    # train_data = utils.CIFAR10Pair(root='data', train=True, transform=utils.train_transform, download=True)
    train_data = CustomDataset(root=args.train_path,transform=utils.train_transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=False,
                              drop_last=False)
    # memory_data = utils.CIFAR10Pair(root='data', train=True, transform=utils.test_transform, download=True)
    memory_data = CustomDataset(root=args.val_path,transform=utils.test_transform)
    memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=False)
    # test_data = utils.CIFAR10Pair(root='data', train=False, transform=utils.test_transform, download=True)
    test_data = CustomDataset(root=args.test_path,transform=utils.test_transform)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=False)

    # model setup and optimizer config
    # model = Model(feature_dim).cuda()
    model = Simclr_ResNet(base_model=args.model_type,feature_dim = feature_dim).cuda()
    flops, params = profile(model, inputs=(torch.randn(1, 3, 128, 128).cuda(),))
    flops, params = clever_format([flops, params])
    print('# Model Params: {} FLOPs: {}'.format(params, flops))
    # optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    optimizer = optim.SGD(model.parameters(), lr=(args.batch_size*0.3)/256, momentum=0.9)
    total_count = args.epochs*len(train_loader)
    # self.optimizer = torch.optim.Adam(params_to_update, lr=self.args.LR)
    scheduler_steplr = CosineAnnealingLR(optimizer,total_count,((args.batch_size*0.3)/256)*(10**(-4)))
    c = len(memory_data.classes)

    # training loop
    results = {'train_loss': [], 'test_acc@1': [], 'test_acc@5': []}
    save_name_pre = '{}_{}_{}_{}_{}'.format(feature_dim, temperature, k, batch_size, epochs)
    if not os.path.exists('unsupervised/SimCLR/results'):
        os.mkdir('unsupervised/SimCLR/results')
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, optimizer,scheduler_steplr)
        results['train_loss'].append(train_loss)
        test_acc_1, test_acc_5 = test(model, memory_loader, test_loader)
        results['test_acc@1'].append(test_acc_1)
        results['test_acc@5'].append(test_acc_5)
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('unsupervised/SimCLR/results/{}_statistics.csv'.format(save_name_pre), index_label='epoch')
        if test_acc_1 > best_acc:
            best_acc = test_acc_1
            torch.save(model.state_dict(), 'unsupervised/SimCLR/results/{}_model.pth'.format(save_name_pre))

