import numpy as np
import cv2
import shutil
import sys
import os
import scipy
import torch
import random
from utils.rotation_conversions import *
import pycocotools.mask as maskUtils
from scipy import signal

def filter_butter(data, N=5, Wn=0.5, btype='low'):
    filterdata = data.copy() #[:,pose_ind].copy()
    b, a = signal.butter(N, Wn, 'low')
    filterdata = signal.filtfilt(b, a, filterdata.T).T.copy()  # butterworth filter

    return filterdata

def annToMask(segm, height, width):
    """
    Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
    :return: binary mask (numpy 2D array)
    """
    def _annToRLE(segm, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = segm
        return rle

    rle = _annToRLE(segm, height, width)
    mask = maskUtils.decode(rle)
    return mask

def seed_worker(worker_seed=7):
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def set_seed(seed):
    # Set a constant random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    g = torch.Generator()
    g.manual_seed(seed)
    return g

def viz_annot(dataset_dir, param, smpl):
    # visulize
    img = cv2.imread(os.path.join(dataset_dir, param['img_path']))
    h = int(img.shape[0] / 100)
    del param['img_path']
    if 'h_w' in param.keys():
        img_h, img_w = param['h_w']
        del param['h_w']
    else:
        img_h, img_w = img.shape[:2]
        
    # render = Renderer(resolution=(img_w, img_h))
    viz_im = img.copy()
    for i in param:
        # viz_im = img.copy()
        if param[i]['betas'] is not None:
            if 'extri' in param[i].keys():
                extri = np.array(param[i]['extri'])
            else:
                extri = np.eye(4)
            intri = np.array(param[i]['intri'])
            beta = torch.from_numpy(np.array(param[i]['betas'])).reshape(-1, 10).to(torch.float32)
            pose = torch.from_numpy(np.array(param[i]['pose'])).reshape(-1, 72).to(torch.float32)
            trans = torch.from_numpy(np.array(param[i]['trans'])).reshape(-1, 3).to(torch.float32)
            verts, joints = smpl(beta, pose, trans)
            # if param[i]['lsp_joints_2d'] is not None:
            #     j2d = np.array(param[i]['lsp_joints_2d'])
            # elif param[i]['halpe_joints_2d_pred'] is not None:
            #     j2d = np.array(param[i]['halpe_joints_2d_pred'])#[self.halpe2lsp]
            _, mesh_2d = surface_project(verts.detach().numpy()[0], extri, intri)
            # mesh_3d, mesh_2d, gt_cam_t = self.wp_project(verts.detach().numpy()[0], joints.detach().numpy()[0], j2d, self.smpl.faces, viz_im, fx=1500., fy=1500.)
            for p in mesh_2d:
                viz_im = cv2.circle(viz_im, tuple(p.astype(np.int)), 1, (0,255,255), -1)
        elif param[i]['halpe_joints_3d'] is not None and param[i]['halpe_joints_3d'].max() > 0:
            if 'extri' in param[i].keys():
                extri = np.array(param[i]['extri'])
            else:
                extri = np.eye(4)
            intri = np.array(param[i]['intri'])
            joints = np.array(param[i]['halpe_joints_3d'])[:,:3]
            _, mesh_2d = surface_project(joints, extri, intri)
            # mesh_3d, mesh_2d, gt_cam_t = self.wp_project(verts.detach().numpy()[0], joints.detach().numpy()[0], j2d, self.smpl.faces, viz_im, fx=1500., fy=1500.)
            for p in mesh_2d:
                viz_im = cv2.circle(viz_im, tuple(p.astype(np.int)), h+3, (0,255,255), -1)

        # if param[i]['halpe_joints_2d'] is not None:
        #     gt_joints = np.array(param[i]['halpe_joints_2d']).reshape(-1, 3)[:,:2]
        #     for p in gt_joints:
        #         viz_im = cv2.circle(viz_im, tuple(p.astype(np.int)), h, (0,0,255), -1)
        #         # vis_img('img', viz_im)
        if param[i]['halpe_joints_2d_pred'] is not None:
            alpha_joints = np.array(param[i]['halpe_joints_2d_pred']).reshape(-1,3)[:,:2]
            for p in alpha_joints:
                viz_im = cv2.circle(viz_im, tuple(p.astype(np.int)), h, (0,255,0), -1)

        # if param[i]['halpe_joints_2d_det'] is not None:
        #     alpha_joints = np.array(param[i]['halpe_joints_2d_det']).reshape(-1,3)[:,:2]
        #     for p in alpha_joints:
        #         viz_im = cv2.circle(viz_im, tuple(p.astype(np.int)), h, (255,0,0), -1)

        # if param[i]['mask_path'] is not None:
        #     mask = cv2.imread(os.path.join(dataset_dir, param[i]['mask_path']), 0)
        #     ratiox = 800/int(mask.shape[0])
        #     ratioy = 800/int(mask.shape[1])
        #     if ratiox < ratioy:
        #         ratio = ratiox
        #     else:
        #         ratio = ratioy
        #     cv2.namedWindow('mask',0)
        #     cv2.resizeWindow('mask',int(mask.shape[1]*ratio),int(mask.shape[0]*ratio))
        #     #cv2.moveWindow(name,0,0)
        #     if mask.max() > 1:
        #         mask = mask/255.
        #     cv2.imshow('mask',mask)
        if param[i]['bbox'] is not None:
            viz_im = cv2.rectangle(viz_im, tuple(np.array(param[i]['bbox'][0], dtype=np.int)), tuple(np.array(param[i]['bbox'][1], dtype=np.int)), color=(255,255,0), thickness=5)
        # # if param[i]['det_bbox'] is not None:
        #     viz_im = cv2.rectangle(viz_im, tuple(np.array(param[i]['det_bbox'][0], dtype=np.int)), tuple(np.array(param[i]['det_bbox'][1], dtype=np.int)), color=(255,0,255), thickness=5)
        #     pass
    cv2.imwrite('test.jpg', viz_im)
    # vis_img('img', viz_im)

def copy(src, dst):
    """copy src to dst"""
    if not os.path.exists(src):
        print("file %s does not exist!!!" %src)
        sys.exit(0)

    folder = os.path.dirname(dst)
    if not os.path.exists(folder):
        os.makedirs(folder)

    shutil.copyfile(src, dst)


def calc_aabb(ptSets):
    ptSets = np.array(ptSets)
    # 筛除 [0, 0] 点
    valid_pts = ptSets[~np.all(ptSets == [0, 0], axis=1)]
    
    if valid_pts.size == 0:
        return np.array([0, 0]), np.array([0, 0])
    
    lt = np.min(valid_pts, axis=0)
    rb = np.max(valid_pts, axis=0)
    
    return lt, rb

def move(src, dst):
    """copy src to dst"""
    if not os.path.exists(src):
        print("file %s does not exist!!!" %src)
        sys.exit(0)

    folder = os.path.dirname(dst)
    if not os.path.exists(folder):
        os.makedirs(folder)

    shutil.move(src, dst)

def get_bbox(verts):
    return np.array([[verts[:,0].min(),verts[:,1].min(),verts[:,2].min()],[verts[:,0].max(),verts[:,1].max(),verts[:,2].max()]])

def load_camera_para(file):
    """"
    load camera parameters
    """
    campose = []
    intra = []
    campose_ = []
    distcoef = []
    intra_ = []
    f = open(file,'r')
    for line in f:
        line = line.strip('\n')
        line = line.rstrip()
        words = line.split()
        if len(words) == 3:
            intra_.append([float(words[0]),float(words[1]),float(words[2])])
        elif len(words) == 4:
            campose_.append([float(words[0]),float(words[1]),float(words[2]),float(words[3])])
        elif len(words) == 5:
            distcoef.append(np.array(words).astype(np.float))
        else:
            pass

    index = 0
    intra_t = []
    for i in intra_:
        index+=1
        intra_t.append(i)
        if index == 3:
            index = 0
            intra.append(intra_t)
            intra_t = []

    index = 0
    campose_t = []
    for i in campose_:
        index+=1
        campose_t.append(i)
        if index == 3:
            index = 0
            campose_t.append([0.,0.,0.,1.])
            campose.append(campose_t)
            campose_t = []
    
    if len(distcoef) == 0:
        distcoef = None

    return np.array(campose), np.array(intra), distcoef

def pair_by_L2_distance(alpha, gt_keps, src_mapper, gt_mapper, dim=17, gt_bbox=None):

    alpha = alpha[:,src_mapper]
    gt_keps = gt_keps[:,gt_mapper]

    alpha = alpha[None,:,:,:]
    gt_keps = gt_keps[:,None,:,:]

    if alpha.shape[-1] == 3:
        pred_conf = alpha[...,-1]
    else:
        pred_conf = np.ones((alpha.shape[0], alpha.shape[1], alpha.shape[2]))
    if gt_keps.shape[-1] == 3:
        gt_conf = gt_keps[...,-1]
    else:
        gt_conf = np.ones((gt_keps.shape[0], gt_keps.shape[1], gt_keps.shape[2]))

    conf = np.sqrt(pred_conf * gt_conf)

    loss = np.linalg.norm(alpha[...,:2] - gt_keps[...,:2], axis=-1) * conf
    loss = np.sum(loss, axis=-1)

    return loss

def matching(gt_joints, pred_2d_pose, gt_mapper, src_mapper, max_people=10):
    """
    input: 
    alpha: 
    """
    gt_content = np.array(gt_joints)
    pred_content = np.array(pred_2d_pose)

    # bbox_size = get_bbox(gt_content)

    # alphapose = pair_by_bbox(pred_2d_pose, gt_joints)
    loss = pair_by_L2_distance(pred_content, gt_content, src_mapper, gt_mapper)
    # loss = loss / bbox_size

    ests_new = []
    bestids = []
    for igt, gt in enumerate(gt_content):
        bestid = np.argmin(loss[igt])
        bestids.append(bestid)
        if loss[igt][bestid] > 100000:
            ests_new.append(None)
        else:
            # print(loss[igt][bestid])
            ests_new.append(pred_2d_pose[bestid])
        loss[:,bestid] = 1e5

    return ests_new, bestids

def convert_world_coord(smpl, pose, trans, shape, extri):
    '''Convert to World Coordinate System'''
    pose = torch.tensor(pose, dtype=torch.float32, device=torch.device('cpu')).reshape(-1, 72)
    trans = torch.tensor(trans, dtype=torch.float32, device=torch.device('cpu')).reshape(-1, 3)
    shape = torch.tensor(shape, dtype=torch.float32, device=torch.device('cpu')).reshape(-1, 10)
    extri = torch.from_numpy(extri).float()

    matrix = torch.linalg.inv(extri.clone())
    f = pose.shape[0]

    zeros_trans = torch.zeros((f, 3), device=pose.device, dtype=pose.dtype)
    verts_zero_cam, joints_zero_cam = smpl(shape, pose, zeros_trans)
    root = joints_zero_cam[:,0,:]

    
    # convert pose params
    oritation = axis_angle_to_matrix(pose[:,:3].clone())
    oritation = torch.matmul(matrix[:3,:3], oritation)
    oritation = matrix_to_axis_angle(oritation)
    pose[:,:3] = oritation


    # # rot root joint
    # root_cam = root + trans
    # root_world = torch.einsum('ij,kj->ki', matrix[:3,:3], root_cam)
    # root_world = root_world + matrix[:3,3]
    # trans = root_world - root

    root_cam = trans
    root_world = torch.einsum('ij,kj->ki', matrix[:3,:3], root_cam)
    root_world = root_world + matrix[:3,3]
    trans = root_world

    pose = pose.detach().cpu().numpy()
    trans = trans.detach().cpu().numpy()

    return pose, trans

def get_rot_trans(campose, photoscan=False):
    trans = []
    rot = []
    for cam in campose:
        # for photoscan parameters
        if photoscan:
            cam = np.linalg.inv(cam)  
        trans.append(cam[:3,3])
        rot.append(cam[:3,:3])
        # rot.append(cv2.Rodrigues(cam[:3,:3])[0])

    return trans, rot

def nomalized(z):
    norm = np.linalg.norm(z)
    z = z / norm
    return z

def fill_nMat(n):
    nMat = np.dot(n.reshape(3,1), n.reshape(1,3))
    nMat = np.eye(3) - nMat
    return nMat

def recompute_3D(points2D, extris, intris):
    num_joint = len(points2D[0])

    AtA = np.zeros((num_joint,3,3))
    Atb = np.zeros((num_joint,1,3))
    skelPos = np.zeros((3,num_joint))

    ts, Rs = get_rot_trans(extris)
    intris_inv = []
    for v in range(len(intris)):
        intris_inv.append(np.linalg.inv(intris[v]))

    for i in range(num_joint):
        for v in range(len(intris)): # 2-views
            keps = points2D[v][i].copy()
            if keps.max() < 1:
                continue
            conf = keps[2].copy()
            keps[2] = 1.0
            intri = intris_inv[v]
            R = Rs[v]
            t = ts[v]
            n = np.dot(intri, keps)
            n = nomalized(n)
            nMat = fill_nMat(n)
            nMat = np.dot(R.T, nMat)
            AtA[i] += np.dot(nMat, R) * (conf + 1e-6)
            Atb[i] += np.dot(-nMat, t) * (conf + 1e-6)
    
    AtA = AtA.astype(np.float32)
    for i in range(num_joint):
        # l, d = LDLT(AtA[i])
        # y = np.linalg.solve(l, Atb[i].T)
        # skelPos[:,i] = np.linalg.solve(d, y).reshape(3,)
        skelPos[:,i] = np.linalg.solve(AtA[i], Atb[i].T).reshape(3,)
        #skelPos.col(i) = AtA[i].ldlt().solve(Atb[i])
    skelPos = skelPos.T
    
    return np.array(skelPos)

def get_transform(center, scale, res, rot=0):
    """Generate transformation matrix."""
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot # To match direction of rotation from cropping
        rot_mat = np.zeros((3,3))
        rot_rad = rot * np.pi / 180
        sn,cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0,:2] = [cs, -sn]
        rot_mat[1,:2] = [sn, cs]
        rot_mat[2,2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0,2] = -res[1]/2
        t_mat[1,2] = -res[0]/2
        t_inv = t_mat.copy()
        t_inv[:2,2] *= -1
        t = np.dot(t_inv,np.dot(rot_mat,np.dot(t_mat,t)))
    return t

def transform(pt, center, scale, res, invert=0, rot=0):
    """Transform pixel location to different reference."""
    t = get_transform(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0]-1, pt[1]-1, 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int)+1

def crop(img, center, scale, res, rot=0):
    """Crop image according to the supplied bounding box."""
    # Upper left point
    ul = np.array(transform([1, 1], center, scale, res, invert=1))-1
    # Bottom right point
    br = np.array(transform([res[0]+1, 
                             res[1]+1], center, scale, res, invert=1))-1
    
    # Padding so that when rotated proper amount of context is included
    pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)
    if not rot == 0:
        ul -= pad
        br += pad

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(new_shape)

    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])
    new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], 
                                                        old_x[0]:old_x[1]]

    if not rot == 0:
        # Remove padding
        new_img = scipy.ndimage.rotate(new_img, rot, reshape=False)
        new_img = new_img[pad:-pad, pad:-pad]

    new_img = cv2.resize(new_img, tuple(res), interpolation=cv2.INTER_CUBIC) #scipy.misc.imresize(new_img, res)
    return new_img

def uncrop(img, center, scale, orig_shape, rot=0, is_rgb=True):
    """'Undo' the image cropping/resizing.
    This function is used when evaluating mask/part segmentation.
    """
    res = img.shape[:2]
    # Upper left point
    ul = np.array(transform([1, 1], center, scale, res, invert=1))-1
    # Bottom right point
    br = np.array(transform([res[0]+1,res[1]+1], center, scale, res, invert=1))-1
    # size of cropped image
    crop_shape = [br[1] - ul[1], br[0] - ul[0]]

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(orig_shape, dtype=np.uint8)
    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], orig_shape[1]) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], orig_shape[0]) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(orig_shape[1], br[0])
    old_y = max(0, ul[1]), min(orig_shape[0], br[1])
    new_img = cv2.resize(img, tuple(crop_shape), interpolation=cv2.INTER_NEAREST)
    new_img[old_y[0]:old_y[1], old_x[0]:old_x[1]] = img[new_y[0]:new_y[1], new_x[0]:new_x[1]]
    return new_img

def save_camparam(path, intris, extris, dist=None):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    f = open(path, 'w')
    for ind, (intri, extri) in enumerate(zip(intris, extris)):
        f.write(str(ind)+'\n')
        for i in intri:
            f.write(str(i[0])+' '+str(i[1])+' '+str(i[2])+'\n')
        if dist is None:
            f.write('0 0 \n')
        else:
            for i in dist[ind]:
                f.write(str(i) + ' ')
            f.write('\n')
        for i in extri[:3]:
            f.write(str(i[0])+' '+str(i[1])+' '+str(i[2])+' '+str(i[3])+'\n')
        f.write('\n')
    f.close()

def rigid_transform_3D(A, B):
    assert len(A) == len(B)

    N = A.shape[0]  # total points
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # centre the points
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))

    H = np.matmul(np.transpose(AA),BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.matmul(Vt.T, U.T)

    # special reflection case
    flag = 0
    if np.linalg.det(R) < 0:
        #print("Reflection detected")
        Vt[2, :] *= -1
        R = np.matmul(Vt.T,U.T)
        flag = 1

    t = -np.matmul(R, centroid_A) + centroid_B
    # err = B - np.matmul(A,R.T) - t.reshape([1, 3])
    return R, t, flag

def surface_project(vertices, exter, intri):
    intri_ = np.insert(intri,3,values=0.,axis=1)
    temp_v = np.insert(vertices,3,values=1.,axis=1).transpose((1,0))
    out_point = np.dot(exter, temp_v)
    mesh_3d = out_point.transpose(1,0)[:,:3]
    dis = out_point[2]
    out_point = (np.dot(intri_, out_point) / dis)[:-1]
    mesh_2d = (out_point.astype(np.int32)).transpose(1,0)
    return mesh_3d, mesh_2d

def draw_keyp(img, joints, color=None, format='coco17', thickness=3):
    skeletons = {'coco17':[[0,1],[1,3],[0,2],[2,4],[5,6],[5,7],[7,9],[6,8],[8,10],[5,11],[11,13],[13,15],[6,12],[12,14],    [14,16],[11,12]],
            'halpe':[[0,1],[1,3],[0,2],[2,4],[5,18],[6,18],[18,17],[5,7],[7,9],[6,8],[8,10],[5,11],[11,13],[13,15],[6,12],[12,14],[14,16],[11,19],[19,12],[18,19],[15,24],[15,20],[20,22],[16,25],[16,21],[21,23]],
            'MHHI':[[0,1],[1,2],[3,4],[4,5],[0,6],[3,6],[6,13],[13,7],[13,10],[7,8],[8,9],[10,11],[11,12]],
            'Simple_SMPL':[[0,1],[1,2],[2,6],[6,3],[3,4],[4,5],[6,7],[7,8],[8,9],[8,10],[10,11],[11,12],[8,13],[13,14],[14,15]],
            'LSP':[[0,1],[1,2],[2,3],[5,4],[4,3],[3,9],[9,8],[8,2],[6,7],[7,8],[9,10],[10,11]],
            }
    colors = {'coco17':[(255,0,0), (255,82,0), (255,165,0), (255,210,0), (255,255,0), (127,255,0), (0,127,0), (0,255,0), (0,210,255), (0,127,255), (0,82,127), (0,210,127), (0,0,127), (0,0,255), (139,0,255), (139,0,127)],
                'halpe':[(255,0,0), (255,82,0), (255,165,0), (255,210,0), (255,255,0), (127,255,0), (0,127,0), (0,255,0), (0,210,255), (0,127,255), (0,82,127), (0,210,127), (0,0,127), (0,0,255), (139,0,255), (139,0,127), (255,0,0), (255,0,0), (255,0,0), (255,0,0), (255,0,0), (255,0,0), (255,0,0), (255,0,0), (255,0,0), (255,0,0), (255,0,0), (255,0,0), (255,0,0), ],
                'MHHI':[(255,0,0), (255,82,0), (255,165,0), (255,210,0), (255,255,0), (127,255,0), (0,127,0), (0,255,0), (0,210,255), (0,127,255), (0,82,127), (0,210,127), (0,0,127)],
                'Simple_SMPL':[(255,0,0), (255,82,0), (255,165,0), (255,210,0), (255,255,0), (127,255,0), (0,127,0), (0,255,0), (0,210,255), (0,127,255), (0,82,127), (0,210,127), (0,0,127), (0,0,127), (0,0,127), (0,0,127), (0,0,127), (0,0,127), (0,0,127), (0,0,127), (0,0,127)],
                'LSP':[(255,0,0), (255,82,0), (255,165,0), (255,210,0), (255,255,0), (127,255,0), (0,127,0), (0,255,0), (0,210,255), (0,127,255), (0,82,127), (0,210,127)]}

    if joints.shape[1] == 3:
        confidence = joints[:,2]
    else:
        confidence = np.ones((joints.shape[0], 1))
    joints = joints[:,:2].astype(np.int32)
    for bone, c in zip(skeletons[format], colors[format]):
        if color is not None:
            c = color
        # c = (0,255,255)
        if confidence[bone[0]] > 0.1 and confidence[bone[1]] > 0.1:
            # pass
            img = cv2.line(img, tuple(joints[bone[0]]), tuple(joints[bone[1]]), c, thickness=int(thickness))
    
    for p in joints:
        img = cv2.circle(img, tuple(p), int(thickness * 5/3), c, -1)
        # vis_img('img', img)
    return img

def vis_img(name, im):
    ratiox = 800/int(im.shape[0])
    ratioy = 800/int(im.shape[1])
    if ratiox < ratioy:
        ratio = ratiox
    else:
        ratio = ratioy

    cv2.namedWindow(name,0)
    cv2.resizeWindow(name,int(im.shape[1]*ratio),int(im.shape[0]*ratio))
    #cv2.moveWindow(name,0,0)
    if im.max() > 1:
        im = im/255.
    cv2.imshow(name,im)
    cv2.waitKey()

def estimate_translation_np(S, joints_2d, joints_conf, fx=5000, fy=5000, cx=128., cy=128.):
    num_joints = S.shape[0]
    # focal length
    f = np.array([fx, fy])
    # optical center
    # center = np.array([img_size/2., img_size/2.])
    center = np.array([cx, cy])
    # transformations
    Z = np.reshape(np.tile(S[:,2],(2,1)).T,-1)
    XY = np.reshape(S[:,0:2],-1)
    O = np.tile(center,num_joints)
    F = np.tile(f,num_joints)
    weight2 = np.reshape(np.tile(np.sqrt(joints_conf),(2,1)).T,-1)

    # least squares
    Q = np.array([F*np.tile(np.array([1,0]),num_joints), F*np.tile(np.array([0,1]),num_joints), O-np.reshape(joints_2d,-1)]).T
    c = (np.reshape(joints_2d,-1)-O)*Z - F*XY

    # weighted least squares
    W = np.diagflat(weight2)
    Q = np.dot(W,Q)
    c = np.dot(W,c)

    # square matrix
    A = np.dot(Q.T,Q)
    b = np.dot(Q.T,c)

    # solution
    trans = np.linalg.solve(A, b)
    return trans

def images_to_video(input_images, output_video, frame_rate):
    
    if isinstance(input_images, str):
        images = [img for img in os.listdir(input_images) if img.endswith(".png") or img.endswith(".jpg")]
        images.sort()  # Sort images by name

        # Read the first image to get the width and height
        frame = cv2.imread(os.path.join(input_images, images[0]))
        height, width, _ = frame.shape
        
        frames = []
        for image in images:
            frame = cv2.imread(os.path.join(input_images, image))
            frames.append(frame)
        
    elif isinstance(input_images, list):
        frames = input_images
        height, width, _ = input_images[0].shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' for .mp4 files
    video = cv2.VideoWriter(output_video, fourcc, frame_rate, (width, height))

    for frame in frames:
        video.write(frame)

    video.release()
    print(f"Video saved as {output_video}")