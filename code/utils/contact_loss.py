import sys
import os

import time
import pickle

import numpy as np

import torch
import torch.nn as nn

import scipy.sparse as sparse

from ChamferDistancePytorch import dist_chamfer as ext
distChamfer = ext.chamferDist()

class GMoF_unscaled(nn.Module):
    def __init__(self, rho=1):
        super(GMoF_unscaled, self).__init__()
        self.rho = rho

    def extra_repr(self):
        return 'rho = {}'.format(self.rho)

    def forward(self, residual):
        squared_res = residual ** 2
        dist = torch.div(squared_res, squared_res + self.rho ** 2)
        return dist

class FootoContactLoss(nn.Module):
    def __init__(self, rho_contact, foot_contact_verts_ids=None, body_segments_dir=None, dtype=torch.float32, use_cuda=True, **kwargs):
        super(FootoContactLoss, self).__init__()
        self.foot_contact_verts_ids = []
        if foot_contact_verts_ids is not None:
            self.foot_contact_verts_ids = foot_contact_verts_ids
        else:
            import json
            with open(os.path.join(body_segments_dir, 'L_Leg' + '.json'), 'r') as f:
                data = json.load(f)
                self.foot_contact_verts_ids.append(list(set(data["verts_ind"])))
            with open(os.path.join(body_segments_dir, 'R_Leg' + '.json'), 'r') as f:
                data = json.load(f)
                self.foot_contact_verts_ids.append(list(set(data["verts_ind"])))
        self.foot_contact_verts_ids = np.concatenate(self.foot_contact_verts_ids)
        self.rho_contact = rho_contact
        self.contact_robustifier = GMoF_unscaled(rho=self.rho_contact)

    def forward(self,body_model_output,foot_contact_verts_ids=None):
        vertices = body_model_output.vertices
        if foot_contact_verts_ids is not None:
            self.foot_contact_verts_ids = foot_contact_verts_ids
        contact_foot_vertices_y = vertices[:, self.foot_contact_verts_ids, 1]
        foot_dist = self.contact_robustifier(contact_foot_vertices_y)
        return foot_dist

class ContactLoss(nn.Module):

    def __init__(self, body_model, scene_path, rho_contact, contact_angle, contact_verts_ids=None, body_segments_dir=None, dtype=torch.float32, use_cuda=True, **kwargs):
        super(ContactLoss, self).__init__()
        CONTACT_BODY_PARTS=['back','gluteus','L_Hand','L_Leg','R_Hand','R_Leg','thighs']
        self.rho_contact = rho_contact
        self.contact_angle = contact_angle
        # self.use_foot_contact = use_foot_contact
        if contact_verts_ids is not None:
            self.contact_verts_ids = contact_verts_ids
        else:
            import json
            self.contact_verts_ids = []
            for part in CONTACT_BODY_PARTS:
                with open(os.path.join(body_segments_dir, part + '.json'), 'r') as f:
                    data = json.load(f)
                    self.contact_verts_ids.append(list(set(data["verts_ind"])))
            self.contact_verts_ids = np.concatenate(self.contact_verts_ids)
            # if self.use_foot_contact:
            #     self.foot_contact_verts_ids = []
            #     with open(os.path.join(body_segments_dir, 'L_Leg' + '.json'), 'r') as f:
            #         data = json.load(f)
            #         self.foot_contact_verts_ids.append(list(set(data["verts_ind"])))
            #     with open(os.path.join(body_segments_dir, 'R_Leg' + '.json'), 'r') as f:
            #         data = json.load(f)
            #         self.foot_contact_verts_ids.append(list(set(data["verts_ind"])))
        
        if use_cuda and torch.cuda.is_available():
            device = torch.device('cuda')
        vertices = body_model(return_verts=True, body_pose= torch.zeros((1, 63), dtype=dtype, device=device)).vertices
        vertices_np = vertices.detach().cpu().numpy().squeeze()
        body_faces_np = np.require(body_model.faces_tensor.detach().cpu().numpy().reshape(-1, 3),dtype=np.uint32)

        ftov = self.faces_by_vertex(vertices_np,body_faces_np,as_sparse_matrix=True)
        ftov = sparse.coo_matrix(ftov)
        indices = torch.LongTensor(np.vstack((ftov.row, ftov.col))).to(device)
        values = torch.FloatTensor(ftov.data).to(device)
        shape = ftov.shape
        self.ftov = torch.sparse.FloatTensor(indices, values, torch.Size(shape))
    
        import open3d as o3d
        scene_mesh = o3d.io.read_triangle_mesh(scene_path)
        scene_mesh.compute_vertex_normals()
        self.scene_vn = torch.tensor(
            np.asarray(scene_mesh.vertex_normals)[np.newaxis,:],
            dtype=dtype,
            device=device)
        self.scene_v = torch.tensor(
            np.asarray(scene_mesh.vertices)[np.newaxis,:],
            dtype=dtype,
            device=device
        )
        self.scene_f = torch.tensor(
            np.asarray(scene_mesh.triangles)[np.newaxis,:].astype(int),
            dtype=torch.long,
            device=device
        )

        self.contact_robustifier = GMoF_unscaled(rho=self.rho_contact)

    def faces_by_vertex(self, v, f, as_sparse_matrix=False):
        # import scipy.sparse as sp
        if not as_sparse_matrix:
            faces_by_vertex = [[] for i in range(len(self.v))]
            for i, face in enumerate(self.f):
                faces_by_vertex[face[0]].append(i)
                faces_by_vertex[face[1]].append(i)
                faces_by_vertex[face[2]].append(i)
        else:
            row = f.flatten()
            col = np.array([range(f.shape[0])]*3).T.flatten()
            data = np.ones(len(col))
            faces_by_vertex = sparse.csr_matrix((data,(row,col)), shape=(v.shape[0],f.shape[0]))
        return faces_by_vertex

    def forward(self,body_model_output,body_model_faces,contact_verts_ids=None,foot_contact_verts_ids=None):
        vertices = body_model_output.vertices
        # body_model_faces = body_model_output.faces.view(-1)
        if contact_verts_ids is not None:
            self.contact_verts_ids = contact_verts_ids
        contact_body_vertices = vertices[:, self.contact_verts_ids, :]
        contact_dist, _, idx1, _ = distChamfer(contact_body_vertices.contiguous(), self.scene_v)
        body_triangles = torch.index_select(
            vertices, 1,
            body_model_faces).view(1, -1, 3, 3)
        edge0 = body_triangles[:, :, 1] - body_triangles[:, :, 0]
        edge1 = body_triangles[:, :, 2] - body_triangles[:, :, 0]
        body_normals = torch.cross(edge0, edge1, dim=2)
        body_normals = body_normals / torch.norm(body_normals, 2, dim=2, keepdim=True)
        body_v_normals = torch.mm(self.ftov, body_normals.squeeze())
        body_v_normals = body_v_normals / torch.norm(body_v_normals, 2, dim=1, keepdim=True)
        contact_body_verts_normals = body_v_normals[self.contact_verts_ids, :]
        contact_scene_normals = self.scene_vn[:, idx1.squeeze().to(dtype=torch.long), :].squeeze()
        angles = torch.asin(torch.norm(torch.cross(contact_body_verts_normals, contact_scene_normals), 2, dim=1, keepdim=True)) *180 / np.pi
        valid_contact_mask = (angles.le(self.contact_angle) + angles.ge(180 - self.contact_angle)).ge(1)
        valid_contact_ids = valid_contact_mask.squeeze().nonzero().squeeze()
        contact_dist = self.contact_robustifier(contact_dist[:, valid_contact_ids].sqrt())

        # if self.use_foot_contact:
        #     if foot_contact_verts_ids is not None:
        #         self.foot_contact_verts_ids = foot_contact_verts_ids
        #     contact_foot_vertices_y = vertices[:, self.foot_contact_verts_ids, 1]
        #     foot_dist = self.contact_robustifier(contact_foot_vertices_y)
        #     return contact_dist, foot_dist
        return contact_dist

'''
class ContactLoss(nn.Module):

    def __init__(self, body_model, scene_path, rho_contact, contact_angle, contact_verts_ids=None, body_segments_dir=None, dtype=torch.float32, use_cuda=True, **kwargs):
        super(ContactLoss, self).__init__()
        CONTACT_BODY_PARTS=['back','gluteus','L_Hand','L_Leg','R_Hand','R_Leg','thighs']
        self.rho_contact = rho_contact
        self.contact_angle = contact_angle
        
        # self.register_buffer('rho_contact',
        #                      torch.tensor(rho_contact, dtype=dtype))
        # self.register_buffer('contact_angle',
        #                      torch.tensor(contact_angle, dtype=dtype))

        if contact_verts_ids is not None:
            self.contact_verts_ids = contact_verts_ids
        else:
            import json
            self.contact_verts_ids = []
            for part in CONTACT_BODY_PARTS:
                with open(os.path.join(body_segments_dir, part + '.json'), 'r') as f:
                    data = json.load(f)
                    self.contact_verts_ids.append(list(set(data["verts_ind"])))
            self.contact_verts_ids = np.concatenate(self.contact_verts_ids)
        
        if use_cuda and torch.cuda.is_available():
            device = torch.device('cuda')
        vertices = body_model(return_verts=True, body_pose= torch.zeros((1, 63), dtype=dtype, device=device)).vertices
        vertices_np = vertices.detach().cpu().numpy().squeeze()
        body_faces_np = np.require(body_model.faces_tensor.detach().cpu().numpy().reshape(-1, 3),dtype=np.uint32)

        ftov = self.faces_by_vertex(vertices_np,body_faces_np,as_sparse_matrix=True)
        ftov = sparse.coo_matrix(ftov)
        indices = torch.LongTensor(np.vstack((ftov.row, ftov.col))).to(device)
        values = torch.FloatTensor(ftov.data).to(device)
        shape = ftov.shape
        self.ftov = torch.sparse.FloatTensor(indices, values, torch.Size(shape))
    
        import open3d as o3d
        scene_mesh = o3d.io.read_triangle_mesh(scene_path)
        scene_mesh.compute_vertex_normals()
        self.scene_vn = torch.tensor(
            np.asarray(scene_mesh.vertex_normals)[np.newaxis,:],
            dtype=dtype,
            device=device)
        self.scene_v = torch.tensor(
            np.asarray(scene_mesh.vertices)[np.newaxis,:],
            dtype=dtype,
            device=device
        )
        self.scene_f = torch.tensor(
            np.asarray(scene_mesh.triangles)[np.newaxis,:].astype(int),
            dtype=torch.long,
            device=device
        )

        self.contact_robustifier = GMoF_unscaled(rho=self.rho_contact)

    def faces_by_vertex(self, v, f, as_sparse_matrix=False):
        # import scipy.sparse as sp
        if not as_sparse_matrix:
            faces_by_vertex = [[] for i in range(len(self.v))]
            for i, face in enumerate(self.f):
                faces_by_vertex[face[0]].append(i)
                faces_by_vertex[face[1]].append(i)
                faces_by_vertex[face[2]].append(i)
        else:
            row = f.flatten()
            col = np.array([range(f.shape[0])]*3).T.flatten()
            data = np.ones(len(col))
            faces_by_vertex = sparse.csr_matrix((data,(row,col)), shape=(v.shape[0],f.shape[0]))
        return faces_by_vertex

    def forward(self,body_model_output,body_model_faces,contact_verts_ids=None):
        vertices = body_model_output.vertices
        # body_model_faces = body_model_output.faces.view(-1)
        if contact_verts_ids is not None:
            self.contact_verts_ids = contact_verts_ids
        contact_body_vertices = vertices[:, self.contact_verts_ids, :]
        contact_dist, _, idx1, _ = distChamfer(contact_body_vertices.contiguous(), self.scene_v)
        body_triangles = torch.index_select(
            vertices, 1,
            body_model_faces).view(1, -1, 3, 3)
        edge0 = body_triangles[:, :, 1] - body_triangles[:, :, 0]
        edge1 = body_triangles[:, :, 2] - body_triangles[:, :, 0]
        body_normals = torch.cross(edge0, edge1, dim=2)
        body_normals = body_normals / torch.norm(body_normals, 2, dim=2, keepdim=True)
        body_v_normals = torch.mm(self.ftov, body_normals.squeeze())
        body_v_normals = body_v_normals / torch.norm(body_v_normals, 2, dim=1, keepdim=True)
        contact_body_verts_normals = body_v_normals[self.contact_verts_ids, :]
        contact_scene_normals = self.scene_vn[:, idx1.squeeze().to(dtype=torch.long), :].squeeze()
        angles = torch.asin(torch.norm(torch.cross(contact_body_verts_normals, contact_scene_normals), 2, dim=1, keepdim=True)) *180 / np.pi
        valid_contact_mask = (angles.le(self.contact_angle) + angles.ge(180 - self.contact_angle)).ge(1)
        valid_contact_ids = valid_contact_mask.squeeze().nonzero().squeeze()
        contact_dist = self.contact_robustifier(contact_dist[:, valid_contact_ids].sqrt())
        return contact_dist
'''