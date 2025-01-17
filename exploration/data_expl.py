import torch
import torchvision
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
from tqdm import tqdm
import torchvision.transforms as transforms
from robustbench.data import load_cifar10
from robustbench.utils import load_model
import numpy as np
import matplotlib.pyplot as plt
from models.cifar10.resnet import ResNet18

def main():
    classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship',
                                'truck')
    transforms_list = [transforms.ToTensor()]
    transforms_test = transforms.Compose(transforms_list)
    x_test, y_true = load_cifar10(n_examples=2500,
                                                data_dir="./data",
                                                transforms_test=transforms_test)
    x_test.to('cpu')
    y_true.to('cpu')
    device = "cpu"
    model_name = 'Wang2023Better_WRN-28-10'#'Wong2020Fast'
    if model_name == "ResNet18":
        model = ResNet18(device)
        model.load_state_dict(torch.load('models/cifar10/resnet18.pt', map_location=device))
    else:
        model = load_model(model_name, dataset="cifar10", threat_model='Linf').to(device)

    y_pred = torch.zeros_like(y_true)

    y = model(x_test.to(device))
    y_pred = y.argmax(dim = 1)
    '''
    for i,x in enumerate(x_test.unsqueeze(0)):
        
        y = model(x.to(device))
        if i == 0:
            print(y.shape)
        y_pred[i]=int(y.argmax().item())
    '''   
    y_pred = y_pred.detach().numpy()

    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
                         columns = [i for i in classes])
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig('output.png')
    
    
if __name__ == "__main__":
    main()

