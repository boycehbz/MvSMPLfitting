# -*- coding: utf-8 -*-

import os, glob
import numpy as np

def expid2model(expr_dir):
    from configer import Configer

    if not os.path.exists(expr_dir): raise ValueError('Could not find the experiment directory: %s' % expr_dir)

    best_model_fname = sorted(glob.glob(os.path.join(expr_dir, 'snapshots', '*.pkl')), key=os.path.getmtime)[-1]
    # try_num = os.path.basename(best_model_fname).split('_')[0]

    print(('Found Trained Model: %s' % best_model_fname))

    # default_ps_fname = glob.glob(os.path.join(expr_dir,'*.ini'))[0]
    # if not os.path.exists(
    #     default_ps_fname): raise ValueError('Could not find the appropriate vposer_settings: %s' % default_ps_fname)
    # ps = Configer(default_ps_fname=default_ps_fname, work_dir = expr_dir, best_model_fname=best_model_fname)

    return best_model_fname

def load_vposer(expr_dir, vp_model='snapshot'):
    '''

    :param expr_dir:
    :param vp_model: either 'snapshot' to use the experiment folder's code or a VPoser imported module, e.g.
    from human_body_prior.train.vposer_smpl import VPoser, then pass VPoser to this function
    :param if True will load the model definition used for training, and not the one in current repository
    :return:
    '''
    import importlib
    import os
    import torch
    from model.VPoser import VPoser

    # settings of Vposer++
    num_neurons = 512
    latentD = 32
    data_shape = [1,23,3]
    trained_model_fname = expid2model(expr_dir)
    
    vposer_pt = VPoser(num_neurons=num_neurons, latentD=latentD, data_shape=data_shape)

    model_dict = vposer_pt.state_dict()
    premodel_dict = torch.load(trained_model_fname).state_dict()
    premodel_dict = {k: v for k ,v in premodel_dict.items() if k in model_dict}
    model_dict.update(premodel_dict)
    vposer_pt.load_state_dict(model_dict)
    print("load pretrain parameters from %s" %trained_model_fname)

    vposer_pt.eval()

    return vposer_pt


def extract_weights_asnumpy(exp_id, vp_model= False):
    from human_body_prior.tools.omni_tools import makepath
    from human_body_prior.tools.omni_tools import copy2cpu as c2c

    vposer_pt, vposer_ps = load_vposer(exp_id, vp_model=vp_model)

    save_wt_dir = makepath(os.path.join(vposer_ps.work_dir, 'weights_npy'))

    weights = {}
    for var_name, var in vposer_pt.named_parameters():
        weights[var_name] = c2c(var)
    np.savez(os.path.join(save_wt_dir,'vposerWeights.npz'), **weights)

    print(('Dumped weights as numpy arrays to %s'%save_wt_dir))
    return vposer_ps, weights

if __name__ == '__main__':
    from human_body_prior.tools.omni_tools import copy2cpu as c2c
    expr_dir = '/ps/project/humanbodyprior/VPoser/smpl/pytorch/0020_06_amass'
    from human_body_prior.train.vposer_smpl import VPoser
    vposer_pt, ps = load_vposer(expr_dir, vp_model='snapshot')
    pose = c2c(vposer_pt.sample_poses(10))
    print(pose.shape)