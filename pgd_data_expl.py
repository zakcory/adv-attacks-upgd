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

from AdvRunner import AdvRunner
from attacks.pgd_attacks import PGD

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
    
    
    model_name = 'Wong2020Fast'#'Wang2023Better_WRN-28-10'#
    if model_name == "ResNet18":
        model = ResNet18(device)
        model.load_state_dict(torch.load('models/cifar10/resnet18.pt', map_location=device))
    else:
        model = load_model(model_name, dataset="cifar10", threat_model='Linf').to(device)

    y_pred = torch.zeros_like(y_true)
    for i in range(10000//250):
        y = model(x_test[i*250:(i+1)*250].to(device))
        y_pred[i*250:(i+1)*250] = y.argmax(dim=1)
    #y = model(x_test.to(device))
    #y_pred = y.argmax(dim = 1)
    '''
    for i,x in enumerate(x_test.unsqueeze(0)):
        
        y = model(x.to(device))
        if i == 0:
            print(y.shape)
        y_pred[i]=int(y.argmax().item())
    '''   
    

    working_on = (y_pred == y_true)
    w_x = x_test[working_on]
    w_y = y_true[working_on]
    print(len(w_x))
    data_RGB_start = [0, 0, 0]
    data_RGB_end = [1, 1, 1]
    data_RGB_offset = [0, 0, 0]
    data_RGB_size = torch.tensor([1, 1, 1]).to(device)
    batch_size = 250
    dtype = w_x.dtype
    data_shape = list(w_x.shape)[1:]
    att_misc_args = {'device': device,
                   'dtype': dtype,
                   'batch_size': batch_size,
                   'data_shape': data_shape,
                   'data_RGB_start': data_RGB_start,
                   'data_RGB_end': data_RGB_end,
                   'data_RGB_size': data_RGB_size,
                   'verbose': False,
                   'report_info': False}

    att_pgd_args = {'norm': 'Linf',
                   'eps': 8/255,
                   'n_restarts': 1,
                   'n_iter': 100,
                   'alpha': 0.01,
                   'rand_init': True}
    
    attack = PGD(model, torch.nn.CrossEntropyLoss(),
                                      misc_args=att_misc_args,
                                      pgd_args=att_pgd_args)
    
    adv_runner = AdvRunner(model, attack, data_RGB_size,
                            device=device, dtype=dtype, verbose=False)
    (init_accuracy, x_adv, y_adv, robust_accuracy, adv_loss,
     acc_steps, avg_loss_steps, perts_max_l_inf,
     adv_batch_compute_time_mean, adv_batch_compute_time_std, tot_adv_compute_time, tot_adv_compute_time_std) = \
        adv_runner.run_standard_evaluation(w_x, w_y, len(w_x), bs=250)
    good_image_index = y_adv!=w_y
    torch.save(good_image_index, "adv_tensor.pt")
    print("percentage: ",good_image_index.sum()/good_image_index.shape[0])
        
        
if __name__ == "__main__":
    main()