from AdvRunner import AdvRunner
import torch
import numpy as np
from tqdm import trange


class UniversalAdvRunner(AdvRunner):
    def __init__(self, model, attack, data_RGB_size, device, dtype, verbose=False):
        super(UniversalAdvRunner, self).__init__(model, attack, data_RGB_size, device, dtype)

    def remove_class(self, x, y, class_index):

        removed_classes = ~y.eq(class_index[0])
        for cl in class_index[1:]:
            removed_classes = torch.logical_and(removed_classes ,~y.eq(cl))

        x_patched = x[removed_classes].clone()
        y_patched = y[removed_classes].clone()
        return x_patched, y_patched

    def run_standard_evaluation(self, x_orig, y_orig, n_examples, bs=250):
        with torch.no_grad():
            orig_device = x_orig.device
            # calculate accuracy
            n_batches = int(np.ceil(n_examples / bs))
            robust_flags, n_robust_examples, init_accuracy = self.run_clean_evaluation(x_orig, y_orig, n_examples, bs, n_batches, orig_device)
            n_robust_batches = int(np.ceil(n_robust_examples / bs))
        self.attack.set_multiplier(targeted=False)

        
        

        x_adv = x_orig[robust_flags].detach().clone()
        y_adv = y_orig[robust_flags].detach().clone()

        # remove classes
        x_adv, y_adv = self.remove_class(x_adv, y_adv, [0, 1, 2, 4, 5, 6, 7, 8, 9])
        n_robust_batches = int(np.ceil(len(y_adv) / bs))


        adv_perts = torch.zeros_like(x_orig).detach()
        adv_perts_loss = torch.zeros(n_examples, dtype=self.dtype, device=orig_device)
        if self.attack_report_info:
            info_shape = [self.attack_restarts, self.attack_iter + 1, n_examples]
            all_succ = torch.zeros(info_shape, dtype=self.dtype, device=orig_device)
            all_loss = torch.zeros(info_shape, dtype=self.dtype, device=orig_device)
        else:
            all_succ = None
            all_loss = None

        for rest in range(self.attack_restarts):
        # with torch.cuda.device(self.device):
        #     start_events = [torch.cuda.Event(enable_timing=True) for batch_idx in range(n_batches)]
        #     end_events = [torch.cuda.Event(enable_timing=True) for batch_idx in range(n_batches)]
            if self.attack.rand_init:
                pert_init = torch.empty((self.attack.data_channels,self.attack.data_w,self.attack.data_h), dtype=self.attack.dtype, device=self.attack.device)\
                .uniform_(-1, 1) * self.attack.eps[0].unsqueeze(0)
                # pert_init = self.attack.project(pert_init)
            else:
                pert_init = torch.zeros_like(x.shape[1:]).unsqueeze(0)

            for k in range(self.attack_iter):
                pert_tmp = pert_init.detach().clone().requires_grad_()

                # test loss and succ for restart and iter and update

                curr_gradient = torch.zeros_like(pert_init)   # initialize gradient

                for batch_idx in trange(n_robust_batches):
                    
                    start_idx = batch_idx * bs
                    end_idx = min((batch_idx + 1) * bs, n_examples)
                    batch_indices = torch.arange(start_idx, end_idx, device=orig_device)
                    x = x_adv[start_idx:end_idx, :].detach().clone().to(self.device)
                    y = y_adv[start_idx:end_idx].detach().clone().to(self.device)

                    # make sure that x is a 4d tensor even if there is only a single datapoint left
                    if len(x.shape) == 3:
                        x.unsqueeze_(dim=0)
                    # start_events[batch_idx].record()
                    curr_gradient += self.attack.perturb(x, y, pert_tmp)
                    # end_events[batch_idx].record()
                    # torch.cuda.empty_cache()
                    # with torch.no_grad():

                    #     batch_x_adv = x + batch_adv_perts
                    #     x_adv[start_idx:end_idx] = batch_x_adv.detach().to(orig_device)
                    #     output = self.model.forward(batch_x_adv)
                    #     batch_y_adv = output.max(dim=1)[1]
                    #     y_adv[start_idx:end_idx] = batch_y_adv.to(orig_device)
                    #     adv_perts[start_idx:end_idx] = batch_adv_perts.detach().to(orig_device)
                    #     adv_perts_loss[start_idx:end_idx] = batch_adv_perts_loss.detach().to(orig_device)
                    #     false_batch = ~y.eq(batch_y_adv).detach().to(orig_device)
                    #     non_robust_indices = batch_indices[false_batch]
                    #     robust_flags[non_robust_indices] = False
                    #     if self.attack_report_info:
                    #         all_succ[:, :, start_idx:end_idx] = all_batch_succ.to(orig_device)
                    #         all_loss[:, :, start_idx:end_idx] = all_batch_loss.to(orig_device)
                avg_graident = curr_gradient / n_examples

                with torch.no_grad():
                    pert_init = self.attack.step(pert_init, avg_graident)   # step
                    eval_loss, succ = self.attack.eval_pert(x_adv[100:350].detach().clone().to('cuda'), y_adv[100:350].detach().clone().to('cuda'), pert_init)   # calculate loss
                    print(f"Epoch {k+1}: Loss={eval_loss.mean()}, Acc={1 - succ.sum().div(len(succ))}")

        

        # verifying L-inf
        perts_max_l_inf = (pert_init.abs() / self.attack.data_RGB_size).view(-1).max(dim=0)[0].item()
        print(f"L-inf Norm: {perts_max_l_inf}")

            
            
            # torch.cuda.synchronize()
            # adv_batch_compute_times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
            # adv_batch_compute_time_mean = np.mean(adv_batch_compute_times)
            # adv_batch_compute_time_std = np.std(adv_batch_compute_times)
            # tot_adv_compute_time = np.sum(adv_batch_compute_times)
            # tot_adv_compute_time_std = np.std([time * n_batches for time in adv_batch_compute_times])

        # with torch.no_grad():
        #     robust_accuracy, adv_loss, perts_max_l_inf = \
        #         self.process_results(n_examples, robust_flags, adv_perts, adv_perts_loss)

        #     if self.verbose:
        #         print("reporting results for adversarial attack: " + self.attack_name)
        #         print("Attack batches runtime mean: " + str(adv_batch_compute_time_mean) + " s")
        #         print("Attack batches runtime std: " + str(tot_adv_compute_time_std) + " s")
        #         print("Attack total runtime: " + str(tot_adv_compute_time) + " s")
        #         print("Attack total runtime std over batches: " + str(tot_adv_compute_time_std) + " s")
        #         print("clean accuracy:")
        #         print(init_accuracy)
        #         print("robust accuracy:")
        #         print(robust_accuracy)
        #         print("perturbations max L_inf:")
        #         print(perts_max_l_inf)
        #         print('nan in tensors: {}, max: {:.5f}, min: {:.5f}'.format(
        #             (adv_perts != adv_perts).sum(), adv_perts.max(),
        #             adv_perts.min()))
        #     if self.attack_report_info:
        #         acc_steps = 1 - (all_succ.sum(dim=2) / n_examples)
        #         avg_loss_steps = all_loss.sum(dim=2) / n_examples
        #     else:
        #         acc_steps = None
        #         avg_loss_steps = None
        #     del all_succ
        #     del all_loss
        #     del adv_perts
        #     torch.cuda.empty_cache()

        #     return init_accuracy, x_adv, y_adv, robust_accuracy, adv_loss, \
        #         acc_steps, avg_loss_steps, perts_max_l_inf, \
        #         adv_batch_compute_time_mean, adv_batch_compute_time_std, tot_adv_compute_time, tot_adv_compute_time_std