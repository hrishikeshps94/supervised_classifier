from matplotlib.font_manager import json_dump
import torch
from model import ResNet,ResNetFuser
import torchvision.transforms as T
from glob import glob
from PIL import Image
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Model load
model = ResNetFuser('resnet18',3,False).to(device)

#Test transforms
test_normal_transform = T.Compose([T.ToTensor(),T.Resize((512,512),interpolation=T.InterpolationMode.BICUBIC),T.Normalize([0.4134], [0.2701])])
test_seg_transform = T.Compose([T.ToTensor(),T.Resize((512,512),interpolation=T.InterpolationMode.BICUBIC),T.Normalize([0.193], [0.392])])
# T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

#Defining class names
class_names = {0:'Normal',1:'NPDR',2:'PDR'}
class_names_out = {0:'P0',1:'P1',2:'P2'}
#Path to test set
main_path = '/mnt/c/Users/Hrishikesh/Desktop/hrishi/WORK/RESEARCH/2022/MCCAI-2022/DRAC/ds/test/DiabeticRetinopathyGrading'
test_path = '/mnt/c/Users/Hrishikesh/Desktop/hrishi/WORK/RESEARCH/2022/MCCAI-2022/DRAC/ds/grading/normal/train/0/'

pred_list  =[]

data = torch.load('checkpoint/kappa.pth')

model.load_state_dict(data['model_state_dict'])
model.eval()
test_image_list = glob(os.path.join(test_path,'*.png'))
f = open(f'{main_path}/SCF.csv','w+')
f.write('case,class,P0,P1,P2'+'\n')


for image_path in test_image_list:
    image_name = image_path.split('/')[-1]
###Image Read ######
    norm_image = Image.open(image_path).convert('L')
    seg_path = image_path.replace('normal','grading_seg')
    seg_image = Image.open(seg_path).convert('L')
    norm_image = test_normal_transform(norm_image)
    seg_image = test_seg_transform(seg_image)
###Model Test######
    with torch.set_grad_enabled(False):
        output = model(norm_image[None,...].to(device),seg_image[None,...].to(device))
    output = torch.nn.functional.softmax(output,dim=-1)
    out_save = output.cpu().numpy()
    accuracy,pred = torch.max(output,dim=-1)
    accuracy = float(accuracy.to('cpu'))
    pred_class_name = class_names[int(pred.to('cpu'))]
    # pred_class_save = class_names_out[int(pred.to('cpu'))]
    pred_class_save = int(pred.to('cpu'))
    f.write(f'{image_name},{pred_class_save},{out_save[...,0][0]},{out_save[...,1][0]},{out_save[...,2][0]}\n')
    print(f'Image {image_name} is a = {pred_class_name} with probability = {accuracy},sum=  {output.sum()}')
f.close()
