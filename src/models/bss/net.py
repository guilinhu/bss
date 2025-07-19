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

        self.conv1d = nn.Conv1d(
            in_channels=in_channel,
            out_channels=De,
            kernel_size=L,
            stride=L // 2,
            padding=L // 2,
        )
        self.activation = nn.GELU()

        self.transconv1d = nn.ConvTranspose1d(
            in_channels=De, out_channels=1, kernel_size=L, stride=L // 2, padding=L // 2, output_padding=0
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

        # print(x.shape) [1, 1, 64000]

        # encoder
        x = self.conv1d(x)  # [B, De, T']
        x = self.activation(x)
        # print("x.shape is {}".format(x.shape)) [1, 256, 8001]
        x = x.transpose(1, 2)  # [B, T', De] [1, 8001, 256]

        if "num_target_speakers" in inputs:
            C = inputs["num_target_speakers"]
            # print("C is {}".format(C))
            # === Separator ===
            sep_out_list, attractor_logits = self.separator(x, C)  # [B, C, T', De]

            # unsure
            B, C, T_, De = sep_out_list[-1].shape
            # print("c is {}".format(C))

            # print("sep out {}".format(sep_out.shape)) [1, 3, 8064, 256]

            # decoder
            # decoder expects shape: [B * C, De, T']
            output_list = []
            for sep_out in sep_out_list:
                sep_out = sep_out.permute(0, 1, 3, 2).contiguous()  # [B, C, De, T']
                sep_out = sep_out.view(B * C, De, T_)

                # print("sep out {}".format(sep_out.shape))[4, 256, 4001]

                waveform = self.transconv1d(sep_out)  # [B * C, 1, T_wav]
                waveform = waveform.view(B, C, -1)  # [B, C, T_wav]

                # print("wav is {}".format(waveform.shape))[1, 4, 32000]

                waveform = waveform[:, :, :T]
                output_list.append(waveform)

            # print("output {}".format(waveform.shape)) [1, 4, 64520]
            # print("logits {}".format(attractor_logits.shape)) [1, 5]

            return {"output": waveform, "attractor_logits": attractor_logits, "output_list": output_list}

        else:
            # === Inference ===
            sep_out_list, attractor_logits = self.separator(x, C=None)  # [B, C_max, T', De]

            probs = self.sigmoid(attractor_logits)  # (B, C_max+1)

            probs = probs[:, :-1]
            estimated_C = (probs > 0.5).sum(dim=1).clamp(min=1)  # (B,) avoid C=0

            # TODO: for now take max estimated speaker from the whole batch
            C = estimated_C.max().item()
            # when C not specified, sep_out contains C_max streams
            sep_out = sep_out[-1][:, :C, :, :]  # Take top-C speakers across batch

            B, C, T_, De = sep_out.shape

            # print("sep out {}".format(sep_out.shape)) [1, 3, 8064, 256]

            # decoder
            # decoder expects shape: [B * C, De, T']
            sep_out = sep_out.permute(0, 1, 3, 2).contiguous()  # [B, C, De, T']
            sep_out = sep_out.view(B * C, De, T_)

            waveform = self.transconv1d(sep_out)  # [B * C, 1, T_wav]
            waveform = waveform.view(B, C, -1)  # [B, C, T_wav]

            waveform = waveform[:, :, :T]

            return {"output": waveform, "attractor_logits": attractor_logits}

    # def compile(self):
    #     self.tfgridnet.forward = torch.compile(self.tfgridnet.forward)
    #     return self


if __name__ == "__main__":
    pass
