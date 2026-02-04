# Load data from different format

import json
import pickle
import numpy as np
import os
import yaml
import joblib

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

def load_yaml(path):
    with open(path, 'r') as f:
        cont = f.read()
        data = yaml.load(cont)
    return data

def load_npz(path):
    data = np.load(path, allow_pickle=True)
    return data

def save_npz(path, data):
    if os.path.isabs(path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
    data = np.savez(path, **data)

def save_npy(path, data):
    data = np.save(path, data)

def load_json(path):
    with open(path) as f:
        param = json.load(f)
    return param

# def load_pkl(path):
#     """"
#     load pkl file
#     """
#     param = pickle.load(open(path, 'rb'),encoding='iso-8859-1')
#     return param

def load_pkl_joblib(path):
    
    param = joblib.load(filename=path, mmap_mode='r')
    # param = pickle.load(open(path, 'rb'),encoding='iso-8859-1')
    return param

def load_pkl(path):
    """"
    load pkl file
    """
    with open(path, 'rb') as f:
        param = pickle.load(f, encoding='iso-8859-1')
    # param = pickle.load(open(path, 'rb'),encoding='iso-8859-1')
    return param

def load_obj(path):
    f = open(path, 'r')
    lines = f.readlines()
    verts, faces = [], []
    for l in lines:
        l = l.rstrip('\n')
        l = l.split(' ')
        if l[0] == 'v':
            verts.append([float(l[1]), float(l[2]),float(l[3]),])
        elif l[0] == 'f':
            try:
                faces.append([int(l[1]), int(l[2]),int(l[3]),])
            except:
                faces.append([int(l[1].split('//')[0]), int(l[2].split('//')[0]),int(l[3].split('//')[0]),])
    return np.array(verts), np.array(faces) - 1

def write_obj(verts, faces, file_name):
    if not os.path.exists(os.path.dirname(file_name)):
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, 'w') as fp:
        for v in verts:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        for f in faces: # To support quadrilateral
            fp.write('f ')
            for i in f:
                fp.write('%d ' %(i+1))
            fp.write('\n')
            # fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

def write_obj_with_color(verts, faces, colors, file_name):
    if not os.path.exists(os.path.dirname(file_name)):
        os.makedirs(os.path.dirname(file_name))
    with open(file_name, 'w') as fp:
        for v, c in zip(verts, colors):
            fp.write('v %f %f %f %f %f %f\n' % (v[0], v[1], v[2], c[0], c[1], c[2]))
        for f in faces + 1:
            fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

# output json
def save_keypoints(pose, name, version=1.1):
    pred_poses = np.asarray(pose, dtype=float)
    out = {
        "version": version,
        "people": []
    }
    for i in range(pred_poses.shape[0]):
        keypoints = pred_poses[i].flatten().tolist()
        person = {
            "pose_keypoints_2d": keypoints
        }
        out["people"].append(person)

    data = load_json('data/keypoints1/0000/Camera00/00001_keypoints.json')

    with open(name, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

def save_pkl(path, result):
    """"
    save pkl file
    """
    folder = os.path.dirname(path)
    if not os.path.exists(folder):
        os.makedirs(folder)

    with open(path, 'wb') as result_file:
        pickle.dump(result, result_file, protocol=2)

def save_json(out_path, data):
    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))
    with open(out_path, 'w') as f:
        json.dump(data, f)
