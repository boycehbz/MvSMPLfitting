import torch
from torch import nn, optim
import numpy as np
from torch.nn import functional as F
import torchgeometry as tgm

class ContinousRotReprDecoder(nn.Module):
    def __init__(self):
        super(ContinousRotReprDecoder, self).__init__()

    def forward(self, module_input):
        reshaped_input = module_input.view(-1, 3, 2)

        b1 = F.normalize(reshaped_input[:, :, 0], dim=1)

        dot_prod = torch.sum(b1 * reshaped_input[:, :, 1], dim=1, keepdim=True)
        b2 = F.normalize(reshaped_input[:, :, 1] - dot_prod * b1, dim=-1)
        b3 = torch.cross(b1, b2, dim=1)

        return torch.stack([b1, b2, b3], dim=-1)

class VPoser(nn.Module):
    def __init__(self, num_neurons, latentD, data_shape, use_cont_repr=True):
        super(VPoser, self).__init__()

        self.latentD = latentD
        self.use_cont_repr = use_cont_repr

        n_features = np.prod(data_shape)
        self.num_joints = data_shape[1]

        self.bodyprior_enc_bn1 = nn.BatchNorm1d(n_features)
        self.bodyprior_enc_fc1 = nn.Linear(n_features, num_neurons)
        self.bodyprior_enc_bn2 = nn.BatchNorm1d(num_neurons)
        self.bodyprior_enc_fc2 = nn.Linear(num_neurons, num_neurons)
        self.bodyprior_enc_mu = nn.Linear(num_neurons, latentD)
        self.bodyprior_enc_logvar = nn.Linear(num_neurons, latentD)
        self.dropout = nn.Dropout(p=.1, inplace=False)

        self.bodyprior_dec_fc1 = nn.Linear(latentD, num_neurons)
        self.bodyprior_dec_fc2 = nn.Linear(num_neurons, num_neurons)

        if self.use_cont_repr:
            self.rot_decoder = ContinousRotReprDecoder()

        self.bodyprior_dec_out = nn.Linear(num_neurons, self.num_joints* 6)

    def encode(self, Pin):
        '''

        :param Pin: Nx(numjoints*3)
        :param rep_type: 'matrot'/'aa' for matrix rotations or axis-angle
        :return:
        '''
        Xout = Pin.view(Pin.size(0), -1)  # flatten input
        Xout = self.bodyprior_enc_bn1(Xout)

        Xout = F.leaky_relu(self.bodyprior_enc_fc1(Xout), negative_slope=.2)
        Xout = self.bodyprior_enc_bn2(Xout)
        Xout = self.dropout(Xout)
        Xout = F.leaky_relu(self.bodyprior_enc_fc2(Xout), negative_slope=.2)
        return torch.distributions.normal.Normal(self.bodyprior_enc_mu(Xout), F.softplus(self.bodyprior_enc_logvar(Xout)))

    def decode(self, Zin, output_type='matrot'):
        assert output_type in ['matrot', 'aa']

        Xout = F.leaky_relu(self.bodyprior_dec_fc1(Zin), negative_slope=.2)
        Xout = self.dropout(Xout)
        Xout = F.leaky_relu(self.bodyprior_dec_fc2(Xout), negative_slope=.2)
        Xout = self.bodyprior_dec_out(Xout)
        if self.use_cont_repr:
            Xout = self.rot_decoder(Xout)
        else:
            Xout = torch.tanh(Xout)

        Xout = Xout.view([-1, 1, self.num_joints, 9])
        if output_type == 'aa': return VPoser.matrot2aa(Xout)
        return Xout

    def forward(self, Pin, input_type='matrot', output_type='matrot'):
        '''

        :param Pin: aa: Nx1xnum_jointsx3 / matrot: Nx1xnum_jointsx9
        :param input_type: matrot / aa for matrix rotations or axis angles
        :param output_type: matrot / aa
        :return:
        '''
        assert output_type in ['matrot', 'aa']
        # if input_type == 'aa': Pin = VPoser.aa2matrot(Pin)
        # if Pin.size(3) == 3: Pin = VPoser.aa2matrot(Pin)
        q_z = self.encode(Pin)
        q_z_sample = q_z.rsample()
        Prec = self.decode(q_z_sample)

        results = {'mean':q_z.mean, 'std':q_z.scale}
        if output_type == 'aa': results['pose_aa'] = VPoser.matrot2aa(Prec)
        else: results['pose_matrot'] = Prec
        return results

    def sample_poses(self, num_poses, output_type='aa', seed=None):
        np.random.seed(seed)
        dtype = self.bodyprior_dec_fc1.weight.dtype
        device = self.bodyprior_dec_fc1.weight.device
        self.eval()
        with torch.no_grad():
            Zgen = torch.tensor(np.random.normal(0., 1., size=(num_poses, self.latentD)), dtype=dtype).to(device)
        return self.decode(Zgen, output_type=output_type)

    @staticmethod
    def matrot2aa(pose_matrot):
        '''
        :param pose_matrot: Nx1xnum_jointsx9
        :return: Nx1xnum_jointsx3
        '''
        batch_size = pose_matrot.size(0)
        homogen_matrot = F.pad(pose_matrot.view(-1, 3, 3), [0,1])
        pose = tgm.rotation_matrix_to_angle_axis(homogen_matrot).view(batch_size, 1, -1, 3).contiguous()
        return pose

    @staticmethod
    def aa2matrot(pose):
        '''
        :param Nx1xnum_jointsx3
        :return: pose_matrot: Nx1xnum_jointsx9
        '''
        batch_size = pose.size(0)
        pose_body_matrot = tgm.angle_axis_to_rotation_matrix(pose.reshape(-1, 3))[:, :3, :3].contiguous().view(batch_size, 1, -1, 9)
        return pose_body_matrot