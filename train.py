from configparser import Interpolation
from pyexpat import model
from statistics import mean
import torch
import os,tqdm
import torch.optim as optim
from dataset import Custom_Dataset
from model import ResNet,ResNetFusion,ResNetFuser
from torch.utils.data import DataLoader
import torchvision.transforms as T
from losses import PolyLoss
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import wandb
from torchmetrics import CohenKappa,AUROC
from sklearn.metrics import roc_auc_score
from metrics import quadratic_weighted_kappa
from torch.optim.lr_scheduler import CosineAnnealingLR
from augmentation import RandomCutmix,RandomMixup
from torchvision.datasets import ImageFolder
import p2t_model
from timm.models import create_model

class Train():
    def __init__(self,args,feature_extract = False) -> None:
        self.args = args
        self.current_epoch = 0
        self.n_epochs  = self.args.epochs
        self.best_kappa = 0
        self.best_auc = 0
        self.feature_extract = feature_extract
        self.data_initaliser()
        self.init_summary()
        self.model_intialiser()
        self.optimizer_loss_initaliser()
        self.cutmix = RandomCutmix(num_classes=3,p=0.9)
        self.mixup = RandomMixup(num_classes=3,p=0.9)

    def init_summary(self):
        wandb.init(project="Classifier",name=self.args.log_name,mode='offline')
        return
    
    def model_intialiser(self):
        self.model = ResNet(self.args.model_type,out_dim=3,feat_extract=self.feature_extract)
        # self.model = ResNetFusion(self.args.model_type,out_dim=3,feat_extract=self.feature_extract)
        # self.model = create_model('p2t_small',pretrained=False,num_classes=3,drop_rate=0.0, drop_path_rate=0.1,drop_block_rate=None,)
        self.model.to(device)
        return
    '''
    T.Normalize([0.4134, 0.4134, 0.4134], [0.2701, 0.2701, 0.2701])
    T.Resize((512,512))
    T.Normalize([0.193, 0.193, 0.193], [0.392, 0.392, 0.392])
    T.Normalize([0.417, 0.417, 0.417], [0.22, 0.22, 0.22])
mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].


    '''
    def data_initaliser(self):
        train_transform = T.Compose([T.Resize((512,512)),T.RandomVerticalFlip(p=0.5),T.RandomHorizontalFlip(p=0.5),T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.5)\
       ,T.RandomAffine((0,360)),T.ToTensor(),T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        val_transform = T.Compose([T.Resize((512,512)),T.ToTensor(),T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        train_loader = ImageFolder(self.args.train_input,transform = train_transform)
        self.train_ds = DataLoader(train_loader, batch_size=self.args.batch_size,shuffle=True, num_workers=os.cpu_count())
        val_loader = ImageFolder(self.args.valid_input,transform = val_transform)
        self.val_ds = DataLoader(val_loader, batch_size=60,shuffle=False, num_workers=os.cpu_count())
        return
    def optimizer_loss_initaliser(self):
        params_to_update = self.model.parameters()
        print("Params to learn:")
        if self.feature_extract:
            params_to_update = []
            for name,param in self.model.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)
                    print("\t",name)
        else:
            for name,param in self.model.named_parameters():
                if param.requires_grad == True:
                    print("\t",name)

        # Observe that all parameters are being optimized
        self.optimizer = optim.SGD(params_to_update, lr=(self.args.batch_size*0.1)/256, momentum=0.9)
        total_count = self.args.epochs*len(self.train_ds)
        # self.optimizer = torch.optim.Adam(params_to_update, lr=self.args.LR)
        # self.scheduler_steplr = CosineAnnealingLR(self.optimizer,total_count,self.args.LR*(10**(-4)))
        self.scheduler_steplr = CosineAnnealingLR(self.optimizer,total_count,((self.args.batch_size*0.1)/256)*(10**(-4)))
        self.criterion = PolyLoss(weight=torch.tensor([0.11,0.325,0.57]).to(device,dtype=torch.float32),reduction='mean')
        # self.criterion = PolyLoss(weight=None,reduction='mean')
        # self.criterion = ACELoss(num_classes=3,feat_dim=)
        self.criterion.to(device)
        self.AUC = AUROC(average='macro',num_classes=3).to(device)
        self.Kappa = CohenKappa(num_classes=3,weights='quadratic').to(device)


    def save_checkpoint(self,type):
        checkpoint_folder = self.args.checkpoint_folder
        if not os.path.exists(checkpoint_folder):
            os.makedirs(checkpoint_folder)
        checkpoint_filename = os.path.join(checkpoint_folder, f'{type}.pth')
        save_data = {
            'step': self.current_epoch,
            'best_auc_roc':self.best_auc,
            'best_kappa':self.best_kappa,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(save_data, checkpoint_filename)


    def load_generator_checkpoint_for_training(self,type='kappa'):
        checkpoint_folder = self.args.checkpoint_folder
        checkpoint_filename = os.path.join(checkpoint_folder, f'{type}.pth')
        if not os.path.exists(checkpoint_filename):
            print("Couldn't find checkpoint file. Starting training from the beginning.")
            return
        data = torch.load(checkpoint_filename)
        self.current_epoch = data['step']
        self.best_auc = data['best_auc_roc']
        self.best_kappa = data['best_kappa']
        self.model.load_state_dict(data['model_state_dict'])
        self.optimizer.load_state_dict(data['optimizer_state_dict'])
        print(f"Restored model at epoch {self.current_epoch}.")

    def train_epoch(self):
        self.model.train()
        for count,(inputs, labels) in enumerate(tqdm.tqdm(self.train_ds)):
            labels = torch.nn.functional.one_hot(labels,num_classes=3)
            if torch.rand((1))[0]>0.1:
                inputs,labels = self.cutmix.forward(inputs, labels.argmax(dim=1))
            else:
                inputs,labels = self.mixup.forward(inputs, labels.argmax(dim=1))
            inputs = inputs.to(device)
            labels = labels.to(device)
            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                with torch.autograd.set_detect_anomaly(True):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs,labels)
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler_steplr.step()
        wandb.log({'Learning rate':self.optimizer.param_groups[0]['lr']})
        print(f'Epoch = {self.current_epoch} Train Loss = {loss.item()}')
        return

    def val_epoch(self):
        self.model.eval()
        sk_auc_roc = 0
        weighted_kappa = 0
        for (inputs, labels) in tqdm.tqdm(self.val_ds):
            inputs = inputs.to(device)
            labels = labels.to(device)
            self.optimizer.zero_grad()
            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)
                outputs = torch.nn.Softmax(dim=1)(outputs)
            self.AUC.update(outputs,labels)
            self.Kappa.update(outputs,labels)
            sk_auc_roc+=roc_auc_score(labels.cpu().numpy(),outputs.cpu().numpy(),average='macro',multi_class='ovo')
            weighted_kappa+=quadratic_weighted_kappa(labels.cpu().numpy(),outputs.argmax(dim=-1).cpu().numpy())
        sk_auc_roc = sk_auc_roc/len(self.val_ds)
        weighted_kappa = weighted_kappa/len(self.val_ds)
        ep_kappa,ep_AUC = self.Kappa.compute(),self.AUC.compute()
        self.AUC.reset()
        self.Kappa.reset()
        wandb.log({'val_kappa':ep_kappa.item()})
        wandb.log({'val_AUC':ep_AUC.item()})
        print(f'Epoch = {self.current_epoch} val_slearn_auc = {sk_auc_roc} val_org_kappa={weighted_kappa} best Kappa = {self.best_kappa},best AUC = {self.best_auc},')
        return ep_kappa,ep_AUC

    def run(self):
        # self.load_generator_checkpoint_for_training()
        for _ in range(self.current_epoch,self.n_epochs-1):
            self.current_epoch+=1
            self.train_epoch()
            curr_kappa,curr_auc = self.val_epoch()
            if curr_kappa>self.best_kappa:
                self.best_kappa = curr_kappa
                self.save_checkpoint(type='kappa')
            if curr_auc>self.best_auc:
                self.best_auc=curr_auc
                self.save_checkpoint(type='auc')      
            self.save_checkpoint(type='last')