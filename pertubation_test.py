import torch
import torchvision
from tqdm import tqdm
import torchvision.transforms as transforms
from robustbench.data import load_cifar10
from robustbench.utils import load_model
from models.cifar10.resnet import ResNet18
import numpy as np

def main():
    classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship',
                                'truck')
    transforms_list = [transforms.ToTensor()]
    transforms_test = transforms.Compose(transforms_list)
    x_test, y_true = load_cifar10(n_examples=10000,
                                                data_dir="./data",
                                                transforms_test=transforms_test)
    device = "cuda"
    x_test = x_test
    y_true = y_true

    epsilon = 8 / 255
    
    
    model_name = 'Wong2020Fast'#'Wang2023Better_WRN-28-10'#
    if model_name == "ResNet18":
        model = ResNet18(device)
        model.load_state_dict(torch.load('models/cifar10/resnet18.pt', map_location=device))
    else:
        model = load_model(model_name, dataset="cifar10", threat_model='Linf').to(device)
    perturb = torch.load("autoattack_pert_Wong2020_mask.pt")
    print("pert shape:",perturb.shape)
    print("pert max value:",(perturb.max()))
    y_pred = torch.zeros_like(y_true)
    y_pred_corrupted = torch.zeros_like(y_true)
    for i in range(int(np.ceil(10000/250))):
        
        y_corrupted = model(x_test[i*250:(i+1)*250].to(device)+perturb)
        
        y_pred_corrupted[i*250:(i+1)*250] = y_corrupted.argmax(dim=1)
    #y = model(x_test.to(device))
    #y_pred = y.argmax(dim = 1)
    print("Robust Accuracy:",(y_pred_corrupted==y_true).sum().div(len(y_true)))
    '''
    for i,x in enumerate(x_test.unsqueeze(0)):
        
        y = model(x.to(device))
        if i == 0:
            print(y.shape)
        y_pred[i]=int(y.argmax().item())
    '''   
    

   
        
        
if __name__ == "__main__":
    main()