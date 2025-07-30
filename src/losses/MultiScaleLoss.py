import torch
import torch.nn as nn
import itertools

# from asteroid.losses.sdr import SingleSrcNegSDR
# from asteroid.losses import pairwise_neg_sisdr

from asteroid.losses.sdr import PairwiseNegSDR
from asteroid.losses.pit_wrapper import PITLossWrapper
import torch.nn.functional as F


def assert_no_allzero_slices(x: torch.Tensor, dim: int = -1):
    zero_mask = x.eq(0).all(dim=dim)

    if zero_mask.any():
        idxs = torch.nonzero(zero_mask, as_tuple=False)
        raise AssertionError(f"all-zero slices at indices {idxs.tolist()}")


class MultiScaleLoss(nn.Module):
    def __init__(self, loss_type="sisdr", recon=1.0, attr=1.0, only_last=False):
        super().__init__()
        self.loss_type = loss_type
        self.only_last = only_last
        # self.sdr_loss = SingleSrcNegSDR(loss_type)  # Supports 'sisdr', 'snr', etc.
        self.pit_wrapper = PITLossWrapper(PairwiseNegSDR(loss_type))
        # self.pit_wrapper = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
        self.bce = nn.BCEWithLogitsLoss()  # For attractor existence loss

        self.recon = recon
        self.attr = attr

    def forward(self, all_est_list, gt, attractors=None, return_perm=False, n_speakers=2):
        # print("all estimate {}".format(all_estimates[0].shape))[4, 4, 64000]
        assert all_est_list[0].shape == gt.shape, f"Mismatch: est shape {all_est_list[0].shape}, gt shape {gt.shape}"

        # zero_tensor = torch.zeros_like(gt)
        assert_no_allzero_slices(gt)

        recon_loss = 0.0
        last_reordered = None
        if not self.only_last:
            for est in all_est_list:
                assert_no_allzero_slices(est)
                assert not torch.isnan(est).any() and not torch.isinf(est).any(), "Found nan/inf in est"
                assert not torch.isnan(gt).any() and not torch.isinf(gt).any(), "Found nan/inf in gt"
                loss, reordered = self.pit_wrapper(est, gt, return_est=True)
                recon_loss += loss
                last_reordered = reordered

            N = len(all_est_list)
            # print("N is {}".format(N))
            recon_loss = recon_loss / N
            # recon_loss = torch.clamp(recon_loss, max=30.0)

        else:
            loss, reordered = self.pit_wrapper(all_est_list[-1], gt, return_est=True)
            # loss = torch.clamp(loss, min=-30.0)
            recon_loss += loss
            last_reordered = reordered

        # assert_no_allzero_slices(all_est_list[-1])
        # loss, reordered = self.pit_wrapper(all_est_list[-1], gt, return_est=True)
        # loss = torch.clamp(loss, min=-30.0)
        # recon_loss += loss
        # last_reordered = reordered

        # attractor loss
        # attractors length C+1
        attr_len = attractors.shape[-1]
        assert attr_len == n_speakers + 1
        # attractors = attractors[:, : n_speakers + 1]
        B, C_plus_1 = attractors.shape
        # create [1 ... 1 0] target
        labels = torch.zeros(B, C_plus_1, device=attractors.device)
        labels[:, : C_plus_1 - 1] = 1
        attractor_loss = self.bce(attractors, labels)

        total_loss = self.recon * recon_loss + self.attr * attractor_loss

        if return_perm:

            # return reordered estimates as well
            return total_loss, recon_loss, attractor_loss, last_reordered

        return total_loss, recon_loss, attractor_loss
