import torch
import torch.nn as nn

import torch.nn.functional as F
import numpy as np

# from src.models.common.dsp import DualWindowTF
from src.models.bss.separator import Separator


# A bss model
class Net(nn.Module):
    def __init__(self, L, De, D, hidden_dim, K, M, n_head, N, in_channel, dropout=0.1, max_spk=4):
        super(Net, self).__init__()

        self.L = L
        self.in_channel = in_channel

        self.conv1d = nn.Conv1d(in_channels=in_channel, out_channels=De, kernel_size=L, stride=L // 2, padding=L // 2)
        self.activation = nn.GELU()

        self.transconv1d = nn.ConvTranspose1d(
            in_channels=De, out_channels=1, kernel_size=L, stride=L // 2, padding=L // 2, output_padding=0, bias=False
        )

        self.separator = Separator(De, D, hidden_dim, K, M, n_head, dropout, N, max_spk)

        self.sigmoid = nn.Sigmoid()

        self.N = N

        # self.stft_module = DualWindowTF(
        #     stft_chunk_size=stft_chunk_size,
        #     stft_back_pad=stft_back_pad,
        #     stft_front_pad=stft_pad_size,
        # )

    def forward(self, inputs):
        """
        mixture: (B, C, T)
        """
        x = inputs["mixture"]
        # print(x.shape)
        B, c, T = x.shape

        # encoder
        x = self.conv1d(x)  # [B, De, T']
        x = self.activation(x)
        # print("x.shape is {}".format(x.shape)) [1, 256, 8001]
        x = x.transpose(1, 2)  # [B, T', De] [1, 8001, 256]

        if "num_target_speakers" in inputs:
            C = inputs["num_target_speakers"]
            # === Separator ===
            # attractor_logits C+1
            sep_out_list, attractor_logits = self.separator(x, C)  # [B, C, T', De]

            # unsure
            B, C_new, T_, De = sep_out_list[-1].shape
            assert C.max().item() == C_new

            # decoder
            # decoder expects shape: [B * C, De, T']
            output_list = []
            for sep_out in sep_out_list:
                sep_out = sep_out.permute(0, 1, 3, 2).contiguous()  # [B, C, De, T']
                sep_out = sep_out.view(B * C, De, T_)

                waveform = self.transconv1d(sep_out)  # [B * C, 1, T_wav]
                waveform = waveform.view(B, C, -1)  # [B, C, T_wav]

                # waveform = waveform[:, :, :T]
                mean = waveform.mean(dim=-1, keepdim=True)
                waveform = waveform - mean
                output_list.append(waveform)

            return {"output": waveform, "attractor_logits": attractor_logits, "output_list": output_list}

        else:
            # === Inference ===
            # attractor_logits length C_max+1
            sep_out_list, attractor_logits = self.separator(x, C=None)  # [B, C_max, T', De]
            # print(sep_out_list[-1].shape)

            probs = self.sigmoid(attractor_logits)  # (B, C_max+1)
            # print(probs)

            probs = probs[:, :-1]

            estimated_C = (probs > 0.5).sum(dim=1).clamp(min=1)  # (B,) avoid C=0
            # TODO: for now take max estimated speaker from the whole batch
            C = estimated_C.max().item()
            # when C not specified, sep_out contains C_max streams
            sep_out = sep_out_list[-1][:, :C, :, :]  # Take top-C speakers across batch

            # mask = probs[0] > 0.5
            # if not mask.any():
            #     best = probs.argmax(dim=1)[0].item()  # scalar idx
            #     mask[best] = True
            # sep_out = sep_out_list[-1][:, mask, :]

            B, C_new, T_, De = sep_out.shape
            assert C == C_new

            # decoder
            # decoder expects shape: [B * C, De, T']
            sep_out = sep_out.permute(0, 1, 3, 2).contiguous()  # [B, C, De, T']
            sep_out = sep_out.view(B * C, De, T_)

            waveform = self.transconv1d(sep_out)  # [B * C, 1, T_wav]
            waveform = waveform.view(B, C, -1)  # [B, C, T_wav]

            # waveform = waveform[:, :, :T]
            mean = waveform.mean(dim=-1, keepdim=True)
            waveform = waveform - mean

            return {"output": waveform, "attractor_logits": attractor_logits}


if __name__ == "__main__":
    pass
