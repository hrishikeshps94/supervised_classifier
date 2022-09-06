import torch
from model import ResNet
import torchvision.transforms as T
from glob import glob
from PIL import Image
import os,tqdm
import numpy as np
from metrics import quadratic_weighted_kappa

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Model load
model = ResNet('resnet18',3).to(device)

#Test transforms
test_transform = T.Compose([T.Resize((512,512)),T.ToTensor()])
# T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

#Defining class names
class_names = {0:'Normal',1:'NPDR',2:'PDR'}
class_names_out = {0:'P0',1:'P1',2:'P2'}
#Path to test set
normal_path = '/mnt/c/Users/Hrishikesh/Desktop/hrishi/WORK/RESEARCH/2022/MCCAI-2022/DRAC/ds/grading/val/0/'
PDR_path = '/mnt/c/Users/Hrishikesh/Desktop/hrishi/WORK/RESEARCH/2022/MCCAI-2022/DRAC/ds/grading/val/1/'
NPDR_path = '/mnt/c/Users/Hrishikesh/Desktop/hrishi/WORK/RESEARCH/2022/MCCAI-2022/DRAC/ds/grading/val/2/'
pred_list  =[]

data = torch.load('checkpoint/classifier_sgd_0.5_aug/kappa.pth')

model.load_state_dict(data['model_state_dict'])
model.eval()
val_input_list = sorted(glob(os.path.join(normal_path,'*.png')))
gt_list  = np.zeros(len(sorted(glob(os.path.join(normal_path,'*.png')))))
val_input_list.extend(sorted(glob(os.path.join(PDR_path,'*.png'))))
gt_list  = np.concatenate([gt_list,np.zeros(len(sorted(glob(os.path.join(PDR_path,'*.png')))))+1])
val_input_list.extend(sorted(glob(os.path.join(NPDR_path,'*.png'))))
gt_list  = np.concatenate([gt_list,np.zeros(len(sorted(glob(os.path.join(NPDR_path,'*.png')))))+2])
pred_list = []
image_list = []
# kappa_score = 0
for image_path in tqdm.tqdm(val_input_list):
    image_name = image_path.split('/')[-1]
###Image Read ######
    image = Image.open(image_path).convert('RGB')
    image = test_transform(image)
    image_list.append(image)
inputs = torch.stack(image_list)
###Model Test######
with torch.set_grad_enabled(False):
    output = model(inputs.to(device))
output = torch.nn.functional.softmax(output,dim=-1)
print(quadratic_weighted_kappa(gt_list,output.argmax(dim=-1).cpu().numpy()))
# gt = gt_list.cpu().numpy()
# accuracy,pred = torch.max(output,dim=-1)
# accuracy = float(accuracy.to('cpu'))
# pred_list.append(pred.cpu().numpy()[0])

# pred_list = np.asarray(pred_list)
# gt_n = np.zeros_like(pred_list)+2
# kappa_score=quadratic_weighted_kappa(np.array([0],dtype=int),pred.cpu().numpy(),min_rating=0, max_rating=2)
# print(kappa_score)
