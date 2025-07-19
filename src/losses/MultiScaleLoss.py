import torch
import torch.nn as nn
import itertools

# from asteroid.losses.sdr import SingleSrcNegSDR
from asteroid.losses.sdr import PairwiseNegSDR
from asteroid.losses.pit_wrapper import PITLossWrapper
import torch.nn.functional as F


class MultiScaleLoss(nn.Module):
    def __init__(self, loss_type="sisdr", recon=0.9, attr=0.1):
        super().__init__()
        self.loss_type = loss_type
        # self.sdr_loss = SingleSrcNegSDR(loss_type)  # Supports 'sisdr', 'snr', etc.
        self.pit_wrapper = PITLossWrapper(PairwiseNegSDR(loss_type))
        self.bce = nn.BCEWithLogitsLoss()  # For attractor existence loss

        self.recon = recon
        self.attr = attr

    def forward(self, all_est_list, gt, attractors=None, return_perm=False, n_speakers=2):
        # print("all estimate {}".format(all_estimates[0].shape))[4, 4, 64000]
        assert all_est_list[0].shape == gt.shape, f"Mismatch: est shape {all_est_list[0].shape}, gt shape {gt.shape}"

        recon_loss = 0.0
        last_reordered = None
        for est in all_est_list:
            # print("est {}".format(est.shape))
            # print("gt {}".format(gt.shape))
            assert not torch.isnan(est).any() and not torch.isinf(est).any(), "Found nan/inf in est"
            assert not torch.isnan(gt).any() and not torch.isinf(gt).any(), "Found nan/inf in gt"
            loss, reordered = self.pit_wrapper(est, gt, return_est=True)
            loss = torch.clamp(loss, min=-30.0)
            recon_loss += loss
            last_reordered = reordered

        N = len(all_est_list)
        # print("N is {}".format(N))
        recon_loss = recon_loss / N

        # attractor loss
        attractors = attractors[:, : n_speakers + 1]
        B, C_plus_1 = attractors.shape
        # create [1 ... 1 0] target
        labels = torch.zeros(B, C_plus_1, device=attractors.device)
        labels[:, : C_plus_1 - 1] = 1
        attractor_loss = self.bce(attractors, labels)

        total_loss = self.recon * recon_loss + self.attr * attractor_loss

        if return_perm:

            # return reordered estimates as well
            return total_loss, last_reordered

        return total_loss
