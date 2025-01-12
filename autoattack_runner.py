import math
import time

import numpy as np
import torch

from autoattack import checks
from autoattack.state import EvaluationState
from attacks.auto_attack.autoattack_base import APGDAttack
from tqdm import tqdm

class logger():
    def __init__(self):
        pass
    def log(str1="",str2="",str3="",str4=""):
        print(str1,str2,str3,str4)
        
class AutoAttack():
    def __init__(self, model, norm='Linf', eps=.3, seed=None, verbose=True,
                 attacks_to_run=[], version='standard', is_tf_model=False,
                 device='cuda', log_path=None):
        self.model = model
        self.norm = norm
        assert norm in ['Linf', 'L2', 'L1']
        self.epsilon = eps
        self.seed = seed
        self.verbose = verbose
        self.attacks_to_run = attacks_to_run
        self.version = version
        self.is_tf_model = is_tf_model
        self.device = device
        self.logger = logger()

        # if version in ['standard', 'plus', 'rand'] and attacks_to_run != []:
        #     raise ValueError("attacks_to_run will be overridden unless you use version='custom'")
        
        if not self.is_tf_model:
            self.apgd = APGDAttack(self.model, n_restarts=5, n_iter=100, verbose=False,
                eps=self.epsilon, norm=self.norm, eot_iter=1, rho=.75, seed=self.seed,
                device=self.device, logger=self.logger)
            
            # from .fab_pt import FABAttack_PT
            # self.fab = FABAttack_PT(self.model, n_restarts=5, n_iter=100, eps=self.epsilon, seed=self.seed,
            #     norm=self.norm, verbose=False, device=self.device)
        
            # from .square import SquareAttack
            # self.square = SquareAttack(self.model, p_init=.8, n_queries=5000, eps=self.epsilon, norm=self.norm,
            #     n_restarts=1, seed=self.seed, verbose=False, device=self.device, resc_schedule=False)
                
            # from .autopgd_base import APGDAttack_targeted
            # self.apgd_targeted = APGDAttack_targeted(self.model, n_restarts=1, n_iter=100, verbose=False,
            #     eps=self.epsilon, norm=self.norm, eot_iter=1, rho=.75, seed=self.seed, device=self.device,
            #     logger=self.logger)
    
        # else:
            # from .autopgd_base import APGDAttack
            # self.apgd = APGDAttack(self.model, n_restarts=5, n_iter=100, verbose=False,
            #     eps=self.epsilon, norm=self.norm, eot_iter=1, rho=.75, seed=self.seed, device=self.device,
            #     is_tf_model=True, logger=self.logger)
            
            # from .fab_tf import FABAttack_TF
            # self.fab = FABAttack_TF(self.model, n_restarts=5, n_iter=100, eps=self.epsilon, seed=self.seed,
            #     norm=self.norm, verbose=False, device=self.device)
        
            # from .square import SquareAttack
            # self.square = SquareAttack(self.model.predict, p_init=.8, n_queries=5000, eps=self.epsilon, norm=self.norm,
            #     n_restarts=1, seed=self.seed, verbose=False, device=self.device, resc_schedule=False)
                
            # from .autopgd_base import APGDAttack_targeted
            # self.apgd_targeted = APGDAttack_targeted(self.model, n_restarts=1, n_iter=100, verbose=False,
            #     eps=self.epsilon, norm=self.norm, eot_iter=1, rho=.75, seed=self.seed, device=self.device,
            #     is_tf_model=True, logger=self.logger)
    
        if version in ['standard', 'plus', 'rand']:
            self.set_version(version)
        
    def get_logits(self, x):
        if not self.is_tf_model:
            return self.model(x)
        else:
            return self.model.predict(x)
    
    def get_seed(self):
        return time.time() if self.seed is None else self.seed
    
    def run_standard_evaluation(self,
                                x_orig,
                                y_orig,
                                n_examples,
                                bs=250,
                                return_labels=False,
                                state_path=None):
        if state_path is not None and state_path.exists():
            state = EvaluationState.from_disk(state_path)
            if set(self.attacks_to_run) != state.attacks_to_run:
                raise ValueError("The state was created with a different set of attacks "
                                 "to run. You are probably using the wrong state file.")
            if self.verbose:
                self.logger.log("Restored state from {}".format(state_path))
                self.logger.log("Since the state has been restored, **only** "
                                "the adversarial examples from the current run "
                                "are going to be returned.")
        else:
            state = EvaluationState(set(self.attacks_to_run), path=state_path)
            state.to_disk()
            if self.verbose and state_path is not None:
                self.logger.log("Created state in {}".format(state_path))                                

        attacks_to_run = list(filter(lambda attack: attack not in state.run_attacks, self.attacks_to_run))
        if self.verbose:
            # self.logger.log('using {} version including {}.'.format(self.version, ', '.join(attacks_to_run)))
            if state.run_attacks:
                self.logger.log('{} was/were already run.'.format(', '.join(state.run_attacks)))

        
        with torch.no_grad():
            # calculate accuracy
            n_batches = int(np.ceil(x_orig.shape[0] / bs))
            if state.robust_flags is None:
                robust_flags = torch.zeros(x_orig.shape[0], dtype=torch.bool, device=x_orig.device)
                y_adv = torch.empty_like(y_orig)
                for batch_idx in range(n_batches):
                    start_idx = batch_idx * bs
                    end_idx = min( (batch_idx + 1) * bs, x_orig.shape[0])

                    x = x_orig[start_idx:end_idx, :].clone().to(self.device)
                    y = y_orig[start_idx:end_idx].clone().to(self.device)
                    output = self.get_logits(x).max(dim=1)[1]
                    y_adv[start_idx: end_idx] = output
                    correct_batch = y.eq(output)
                    robust_flags[start_idx:end_idx] = correct_batch.detach().to(robust_flags.device)

                state.robust_flags = robust_flags
                robust_accuracy = torch.sum(robust_flags).item() / x_orig.shape[0]
                robust_accuracy_dict = {'clean': robust_accuracy}
                state.clean_accuracy = robust_accuracy
                
                if self.verbose:
                    self.logger.log('initial accuracy: {:.2%}'.format(robust_accuracy))
            else:
                robust_flags = state.robust_flags.to(x_orig.device)
                robust_accuracy = torch.sum(robust_flags).item() / x_orig.shape[0]
                robust_accuracy_dict = {'clean': state.clean_accuracy}
                if self.verbose:
                    self.logger.log('initial clean accuracy: {:.2%}'.format(state.clean_accuracy))
                    self.logger.log('robust accuracy at the time of restoring the state: {:.2%}'.format(robust_accuracy))
                    
            x_adv = x_orig.clone().detach()
            startt = time.time()
            for attack in attacks_to_run:
                # item() is super important as pytorch int division uses floor rounding
                num_robust = torch.sum(robust_flags).item()

                if num_robust == 0:
                    break

                n_batches = int(np.ceil(num_robust / bs))

                robust_lin_idcs = torch.nonzero(robust_flags, as_tuple=False)
                if num_robust > 1:
                    robust_lin_idcs.squeeze_()
                
                # for batch_idx in range(n_batches):
                #     start_idx = batch_idx * bs
                #     end_idx = min((batch_idx + 1) * bs, num_robust)

                #     batch_datapoint_idcs = robust_lin_idcs[start_idx:end_idx]
                #     if len(batch_datapoint_idcs.shape) > 1:
                #         batch_datapoint_idcs.squeeze_(-1)
                #     x = x_orig[batch_datapoint_idcs, :].clone().to(self.device)
                #     y = y_orig[batch_datapoint_idcs].clone().to(self.device)

                #     # make sure that x is a 4d tensor even if there is only a single datapoint left
                #     if len(x.shape) == 3:
                #         x.unsqueeze_(dim=0)
                    
                # run attack
                if attack == 'apgd-ce':
                    # apgd on cross-entropy loss
                    self.apgd.loss = 'ce'
                    self.apgd.seed = self.get_seed()

                    delta = torch.load('autoattack_pert_Wong2020.pt')

                    # adv_curr, acc, loss_best, x_best_adv = self.apgd.attack_single_run(x_orig.clone()[robust_flags].to(self.device)\
                    #                                                     , y_orig.clone()[robust_flags], n_batches) 
                    # torch.save(adv_curr, 'pert_img_Wang2023Better.pt')
                    # adv_curr = torch.load('pert_img_Wong2020.pt')
                
                elif attack == 'apgd-dlr':
                    # apgd on dlr loss
                    self.apgd.loss = 'dlr'
                    self.apgd.seed = self.get_seed()
                    adv_curr = self.apgd.perturb(x, y) #cheap=True
                
                else:
                    raise ValueError('Attack not supported')
                # print("shapes: ",x_orig.shape,adv_curr.shape)
                # diff = (adv_curr.to(self.device)-x_orig[robust_flags].to(self.device))
                # delta = diff.abs().max(dim = 0)[0]*torch.sign(diff.mean(dim = 0)).clamp_(-self.epsilon, self.epsilon)

                delta.clamp_(-self.epsilon, self.epsilon)
                torch.save(delta, 'autoattack_pert_Wong2020.pt')
                print("delta shape:",delta.shape)
                print("delta max value:",(delta.max()))
                print(f"epsilon flag:{delta.max() <= self.epsilon}")

                bs = 250

                n_batches = int(np.ceil(8334 / bs))

                y_adv = torch.zeros_like(y_orig[robust_flags]).to(self.device)

                for batch_idx in tqdm(range(n_batches)):
                    start_idx = batch_idx * bs
                    end_idx = min( (batch_idx + 1) * bs, x_orig.shape[0])

                    x = x_orig[robust_flags][start_idx:end_idx, :].clone().to(self.device)
                    output = self.get_logits(x + delta).max(dim=1)[1]
                    y_adv[start_idx: end_idx] = output
                
                adv_accuracy = (y_adv == y_orig[robust_flags].to(self.device)).sum().div(len(y_orig))
                print(f"Adv. Achieved Accuracy: {adv_accuracy}")

                # torch.save(delta, 'autoattack_pert_Wong2020.pt')
                    


                if self.verbose:
                    num_non_robust_batch = torch.sum(false_batch)    
                    self.logger.log('{} - {}/{} - {} out of {} successfully perturbed'.format(
                        attack, batch_idx + 1, n_batches, num_non_robust_batch, x.shape[0]))
                
                robust_accuracy = torch.sum(robust_flags).item() / x_orig.shape[0]
                robust_accuracy_dict[attack] = robust_accuracy
                state.add_run_attack(attack)
                if self.verbose:
                    self.logger.log('robust accuracy after {}: {:.2%} (total time {:.1f} s)'.format(
                        attack.upper(), robust_accuracy, time.time() - startt))
                    
            # check about square
            checks.check_square_sr(robust_accuracy_dict, logger=self.logger)
            state.to_disk(force=True)
            
            # final check
            if self.verbose:
                if self.norm == 'Linf':
                    res = (x_adv - x_orig).abs().reshape(x_orig.shape[0], -1).max(1)[0]
                elif self.norm == 'L2':
                    res = ((x_adv - x_orig) ** 2).reshape(x_orig.shape[0], -1).sum(-1).sqrt()
                elif self.norm == 'L1':
                    res = (x_adv - x_orig).abs().reshape(x_orig.shape[0], -1).sum(dim=-1)
                self.logger.log('max {} perturbation: {:.5f}, nan in tensor: {}, max: {:.5f}, min: {:.5f}'.format(
                    self.norm, res.max(), (x_adv != x_adv).sum(), x_adv.max(), x_adv.min()))
                self.logger.log('robust accuracy: {:.2%}'.format(robust_accuracy))
        if return_labels:
            return x_adv, y_adv
        else:
            return x_adv
        
    def clean_accuracy(self, x_orig, y_orig, bs=250):
        n_batches = math.ceil(x_orig.shape[0] / bs)
        acc = 0.
        for counter in range(n_batches):
            x = x_orig[counter * bs:min((counter + 1) * bs, x_orig.shape[0])].clone().to(self.device)
            y = y_orig[counter * bs:min((counter + 1) * bs, x_orig.shape[0])].clone().to(self.device)
            output = self.get_logits(x)
            acc += (output.max(1)[1] == y).float().sum()
            
        if self.verbose:
            print('clean accuracy: {:.2%}'.format(acc / x_orig.shape[0]))
        
        return acc.item() / x_orig.shape[0]
        
    def run_standard_evaluation_individual(self, x_orig, y_orig, bs=250, return_labels=False):
        if self.verbose:
            print('using {} version including {}'.format(self.version,
                ', '.join(self.attacks_to_run)))
        
        l_attacks = self.attacks_to_run
        adv = {}
        verbose_indiv = self.verbose
        self.verbose = False
        
        for c in l_attacks:
            startt = time.time()
            self.attacks_to_run = [c]
            x_adv, y_adv = self.run_standard_evaluation(x_orig, y_orig, bs=bs, return_labels=True)
            if return_labels:
                adv[c] = (x_adv, y_adv)
            else:
                adv[c] = x_adv
            if verbose_indiv:    
                acc_indiv  = self.clean_accuracy(x_adv, y_orig, bs=bs)
                space = '\t \t' if c == 'fab' else '\t'
                self.logger.log('robust accuracy by {} {} {:.2%} \t (time attack: {:.1f} s)'.format(
                    c.upper(), space, acc_indiv,  time.time() - startt))
        
        return adv
        
    def set_version(self, version='standard'):
        if self.verbose:
            print('setting parameters for {} version'.format(version))
        
        if version == 'standard':
            self.attacks_to_run = ['apgd-ce', 'apgd-t', 'fab-t', 'square']
            if self.norm in ['Linf', 'L2']:
                self.apgd.n_restarts = 1
                self.apgd_targeted.n_target_classes = 9
            elif self.norm in ['L1']:
                self.apgd.use_largereps = True
                self.apgd_targeted.use_largereps = True
                self.apgd.n_restarts = 5
                self.apgd_targeted.n_target_classes = 5
            self.fab.n_restarts = 1
            self.apgd_targeted.n_restarts = 1
            self.fab.n_target_classes = 9
            #self.apgd_targeted.n_target_classes = 9
            self.square.n_queries = 5000
        
        elif version == 'plus':
            self.attacks_to_run = ['apgd-ce', 'apgd-dlr', 'fab', 'square', 'apgd-t', 'fab-t']
            self.apgd.n_restarts = 5
            self.fab.n_restarts = 5
            self.apgd_targeted.n_restarts = 1
            self.fab.n_target_classes = 9
            self.apgd_targeted.n_target_classes = 9
            self.square.n_queries = 5000
            if not self.norm in ['Linf', 'L2']:
                print('"{}" version is used with {} norm: please check'.format(
                    version, self.norm))
        
        elif version == 'rand':
            self.attacks_to_run = ['apgd-ce', 'apgd-dlr']
            self.apgd.n_restarts = 1
            self.apgd.eot_iter = 20