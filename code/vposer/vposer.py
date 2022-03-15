from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os.path as osp
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as torchdist
import torchgeometry as tgm


class ContinousRotReprDecoder(nn.Module):
    def __init__(self):
        super(ContinousRotReprDecoder, self).__init__()

    def forward(self, module_input):
        reshaped_input = module_input.view(-1, 3, 2)

        b1 = F.normalize(reshaped_input[:, :, 0], dim=1)

        dot_prod = torch.sum(b1 * reshaped_input[:, :, 1],
                             dim=1, keepdim=True)
        b2 = F.normalize(reshaped_input[:, :, 1] -
                         dot_prod * b1, dim=-1)
        b3 = torch.cross(b1, b2, dim=1)

        return torch.stack([b1, b2, b3], dim=-1)


class VPoserEncoder(nn.Module):

    def __init__(self, latent_dim_size=32, hidden_dim=512, is_training=False,
                 dtype=torch.float32, encoder_ckpt='',
                 **kwargs):
        super(VPoserEncoder, self).__init__()

        self.dtype = dtype

        self.is_training = is_training
        self.fc1 = nn.Linear(23 * 9, 512)
        self.fc2 = nn.Linear(512, 512)
        self.Zmu = nn.Linear(512, latent_dim_size)
        self.Zsigma = nn.Linear(512, latent_dim_size)
        self.latent_dim_size = latent_dim_size

        if encoder_ckpt:
            if encoder_ckpt.endswith('.npy') or encoder_ckpt.endswith('.npz'):
                self.load_from_numpy(encoder_ckpt)
            elif encoder_ckpt.endswith('.pt'):
                data = torch.load(encoder_ckpt)

                num_fc = 3
                enc_weights = {}
                for key in sorted(data):
                    if 'enc' not in key:
                        continue
                    dest_key = (key if 'out' not in key else
                                key.replace('out', 'fc{}'.format(num_fc)))
                    dest_key = dest_key.replace('bodyprior_enc_', '')
                    enc_weights[dest_key] = data[key]

                self.load_state_dict(enc_weights)
        if not self.is_training:
            self.eval()

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        fc1_out = F.leaky_relu(self.fc1(x), 0.2)
        fc1_out = F.dropout(fc1_out, p=0.25, training=self.is_training)
        fc2_out = F.leaky_relu(self.fc2(fc1_out), 0.2)
        Zmu_op = self.Zmu(fc2_out)

        if self.is_training:
            Zsigma_op = F.softplus(self.Zsigma(fc2_out))
            return torchdist.Normal(loc=Zmu_op, scale=Zsigma_op)
        else:
            return Zmu_op

    def load_from_numpy(self, file_path):
        import numpy as np
        data = np.load(file_path)
        # TF data is 32X512 while pytorch layer data has format 512X32
        # that is why transpose
        self.fc1.weight.data = torch.from_numpy(data['fc1W'].T).to(self.dtype)
        self.fc1.bias.data = torch.from_numpy(data['fc1b']).to(self.dtype)
        self.fc2.weight.data = torch.from_numpy(data['fc2W'].T).to(self.dtype)
        self.fc2.bias.data = torch.from_numpy(data['fc2b']).to(self.dtype)
        self.Zmu.weight.data = torch.from_numpy(data['ZmuW'].T).to(self.dtype)
        self.Zmu.bias.data = torch.from_numpy(data['Zmub']).to(self.dtype)


class VPoserDecoder(nn.Module):

    def __init__(self, latent_dim_size=32, is_training=False,
                 num_neurons=512,
                 dtype=torch.float32, vposer_ckpt='',
                 vposer_ignore_hands=True,
                 num_fc=3, data_dim=23 * 9,
                 use_cont_repr=False,
                 **kwargs):

        super(VPoserDecoder, self).__init__()
        self.dtype = dtype
        self.is_training = is_training
        self.latent_dim_size = latent_dim_size
        self.ignore_hands = vposer_ignore_hands
        self.num_fc = num_fc

        self.fc1 = nn.Linear(latent_dim_size, num_neurons)
        self.fc2 = nn.Linear(num_neurons, num_neurons)
        input_dim = latent_dim_size
        output_dim = num_neurons
        self.use_cont_repr = use_cont_repr

        if self.use_cont_repr:
            self.rot_decoder = ContinousRotReprDecoder()

        n_features = data_dim
        if use_cont_repr:
            rot_dim = 3
            num_joints = int(data_dim / 9)
            n_features = int(num_joints * (rot_dim ** 2 - rot_dim))

        for fc_id in range(self.num_fc):
            self.add_module('fc{}'.format(fc_id + 1),
                            nn.Linear(input_dim, output_dim))
            input_dim = output_dim
            output_dim = num_neurons if fc_id < num_fc - 2 else n_features

        if vposer_ckpt:
            self.load_weights(vposer_ckpt)

    def get_dim_size(self):
        return self.latent_dim_size

    def get_mean(self):
        return torch.zeros([self.latent_dim_size],
                           dtype=self.dtype,
                           device=self.fc1.weight.data.device)

    def forward(self, x):
        batch_size = x.shape[0]


        net_input = x
        for fc_id in range(self.num_fc - 1):
            curr_module = getattr(self, 'fc{}'.format(fc_id + 1))
            dropout = F.dropout if fc_id == 1 else lambda arg, **kwargs: arg
            curr_output = dropout(
                F.leaky_relu(curr_module(net_input), 0.2), p=0.25,
                training=self.is_training)
            net_input = curr_output

        curr_module = getattr(self, 'fc{}'.format(self.num_fc))

        output = curr_module(net_input)
        if self.use_cont_repr:
            correct_rot = self.rot_decoder(output)
        else:
            output = torch.tanh(output).view(-1, 3, 3)

            # Before converting the output rotation matrices of the VAE to
            # axis-angle representation, we first need to make them in to valid
            # rotation matrices
            with torch.no_grad():
                # Iterate over the batch dimension and compute the SVD
                svd_input = output.detach().cpu()
                #  svd_input = output
                norm_rotation = torch.zeros_like(svd_input)
                for bidx in range(output.shape[0]):
                    U, _, V = torch.svd(svd_input[bidx])

                    # Multiply the U, V matrices to get the closest orthonormal
                    # matrix
                    norm_rotation[bidx] = torch.matmul(U, V.t())
                norm_rotation = norm_rotation.to(x.device)

            # torch.svd supports backprop only for full-rank matrices.
            # The output is calculated as the valid rotation matrix plus the
            # output minus the detached output. If one writes down the
            # computational graph for this operation, it will become clear the
            # output is the desired valid rotation matrix, while for the
            #  backward pass gradients are propagated only to the original
            # matrix
            # Source: PyTorch Gumbel-Softmax hard sampling
            # https://pytorch.org/docs/stable/_modules/torch/nn/functional.html#gumbel_softmax
            correct_rot = norm_rotation - output.detach() + output
        output = tgm.rotation_matrix_to_angle_axis(
            F.pad(correct_rot.view(-1, 3, 3), [0, 1, 0, 0])).contiguous().view(batch_size,-1)
        if self.ignore_hands:
            return output[:, :-6]
        else:
            return output

    def load_weights(self, file_path):
        file_path = osp.expanduser(osp.expandvars(file_path))
        if file_path.endswith('.npy') or file_path.endswith('.npz'):
            self.load_from_numpy(file_path)
        elif file_path.endswith('.pt'):
            data = torch.load(file_path)

            dec_weights = {}
            for key in sorted(data):
                if 'dec' not in key:
                    continue
                dest_key = (key if 'out' not in key else
                            key.replace('out', 'fc{}'.format(self.num_fc)))
                dest_key = dest_key.replace('bodyprior_dec_', '')
                dec_weights[dest_key] = data[key]

            self.load_state_dict(dec_weights)

    def load_from_numpy(self, file_path):
        import numpy as np
        data = np.load(file_path)
        # TF data is 32X512 while pytorch layer data has format 512X32
        # that is why transpose
        self.fc1.weight.data = torch.from_numpy(data['fc1W'].T).to(self.dtype)
        self.fc1.bias.data = torch.from_numpy(data['fc1b']).to(self.dtype)
        self.fc2.weight.data = torch.from_numpy(data['fc2W'].T).to(self.dtype)
        self.fc2.bias.data = torch.from_numpy(data['fc2b']).to(self.dtype)
        self.fc3.weight.data = torch.from_numpy(data['fc3W'].T).to(self.dtype)
        self.fc3.bias.data = torch.from_numpy(data['fc3b']).to(self.dtype)
