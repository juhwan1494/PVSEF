import os  
import torch  
import torch.nn as nn  
import torch.nn.functional as F  
import numpy as np  
from torch.utils.data import Dataset, DataLoader  
from transformers import VivitForVideoClassification  
import glob  
import cv2  
from pathlib import Path  


n_classes = 2 
num_workers = 8  
CAM_SAVE_DIR = "/your_directory/CAM/Macro/" 

# Grad-CAM 클래스 정의
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model  
        self.target_layer = target_layer  
        self.gradients = None  
        self.activations = None  
        
        self.target_layer.register_forward_hook(self.save_activations) 
        self.target_layer.register_full_backward_hook(self.save_gradients)  

    def save_activations(self, module, input, output):
        self.activations = output  

    def save_gradients(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]  

    def generate(self, x, class_idx):
        self.model.eval()  
        output = self.model(x)  
        
        self.model.zero_grad()  
        class_score = output[0, class_idx]  
        class_score.backward() 

        gradients = self.gradients  
        activations = self.activations  
        
        weights = torch.mean(gradients, dim=(2, 3, 4), keepdim=True)  
        
        cam = torch.sum(weights * activations, dim=1) 
        cam = F.relu(cam)  
        
        cam = F.interpolate(cam.unsqueeze(1), size=(32, 224, 224), mode='trilinear', align_corners=False) 
        cam = cam.squeeze(1) 
        
        cam = cam - cam.min()  
        cam = cam / (cam.max() + 1e-8) 
        
        return cam.cpu().detach().numpy()  

def save_cam(cam, video_frames, patient_name, patch_number, fold_n, save_dir):
    save_dir = os.path.join(save_dir, f"{fold_n}fold") 
    os.makedirs(save_dir, exist_ok=True) 
    
    for t in range(cam.shape[1]):  
        cam_frame = cam[0, t]  
        cam_frame = (cam_frame * 255).astype(np.uint8) 
        
        heatmap = cv2.applyColorMap(cam_frame, cv2.COLORMAP_JET) 
        
        frame = video_frames[t].permute(1, 2, 0).cpu().numpy()  
        frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-8) 
        frame = (frame * 255).astype(np.uint8)  
        
        overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0.0) 
        
        save_path = os.path.join(save_dir, f"{patient_name}_{patch_number}_{t+1}.png")
        cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))  

def get_test_files(fold_n):
    validation_files = [] 
    print("Get test files.")  
    for label in ['0', '1']: 
        dir_path = glob.glob(f'/your_directory/Macro/{fold_n}fold/test/{label}/*/*')  
        for file in dir_path:
            label_vector = [1, 0] if label == '0' else [0, 1]  
            validation_files.append((file, label_vector))  

    print(f'testset : {len(validation_files)}')  
    return validation_files 

class IntracranialDataset(Dataset):
    def __init__(self, df, mode='test'):
        self.data = df  
        self.mode = mode  

    def __len__(self):
        return len(self.data) 

    def __getitem__(self, idx):
        img_name = self.data[idx][0]  
        video = torch.tensor(np.load(img_name))  
        labels = torch.tensor(self.data[idx][1])  
        return {'image': video, 'labels': labels, 'path': img_name}  

class MYVIVIT(nn.Module):
    def __init__(self, n_classes):
        super(MYVIVIT, self).__init__() 
        self.model = VivitForVideoClassification.from_pretrained("google/vivit-b-16x2-kinetics400")  
        self.relu = nn.ReLU()  
        self.fc = nn.Linear(400, n_classes)  

    def forward(self, x):
        x = self.model(x) 
        x = self.fc(x.logits)  
        return x  

def generate_cam(model, dataLoader, fold_n):
    model.eval() 
    grad_cam = GradCAM(model, model.module.model.vivit.embeddings.patch_embeddings.projection)  

    for i, batch in enumerate(dataLoader):
        input = batch["image"].float().cuda()  
        img_path = batch["path"]  

        file_name = Path(img_path[0]).name  
        patient_name = file_name.split('_')[0]  
        patch_number = file_name.split('_')[1].split('.')[0]  

        cam = grad_cam.generate(input, class_idx=1)  
        
        
        save_dir = CAM_SAVE_DIR  
        save_cam(cam, input[0], patient_name, patch_number, fold_n, save_dir)  

        del input  

for fold_n in [1, 2, 3, 4, 5]:  
    savedir = f'/your_directory/Macro/{fold_n}fold/'  
    os.makedirs(savedir, exist_ok=True)  
    print(f'Processing fold {fold_n}')  

    net = MYVIVIT(n_classes) 
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net).cuda()  
    checkpoint = torch.load(f'/your_directory/{fold_n}fold.pth') 
    net.load_state_dict(checkpoint['model_state_dict'])  

    testdata = get_test_files(fold_n)  
    valdataset = IntracranialDataset(testdata, mode='test')  
    valloader = DataLoader(valdataset, batch_size=1, shuffle=False, num_workers=num_workers, drop_last=False)  

    generate_cam(net, valloader, fold_n)  
    
    del net  