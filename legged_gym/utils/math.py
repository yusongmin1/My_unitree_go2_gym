import torch
from torch import Tensor
import numpy as np
from isaacgym.torch_utils import quat_apply, normalize, quat_conjugate, quat_mul
from typing import Tuple

# @ torch.jit.script
def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)

# @ torch.jit.script
def wrap_to_pi(angles):
    angles %= 2*np.pi
    angles -= 2*np.pi * (angles > np.pi)
    return angles

# @ torch.jit.script
def torch_rand_sqrt_float(lower, upper, shape, device):
    # type: (float, float, Tuple[int, int], str) -> Tensor
    r = 2*torch.rand(*shape, device=device) - 1
    r = torch.where(r<0., -torch.sqrt(-r), torch.sqrt(r))
    r =  (r + 1.) / 2.
    return (upper - lower) * r + lower


# @torch.jit.script
def quat_distance(q1, q2,as_quat=False):
    q1_conj = quat_conjugate(q1)

    if as_quat:
        return quat_log(quat_mul(q1_conj, q2))
    else:

        dist = 2*torch.linalg.norm(quat_log(quat_mul(q1_conj, q2)),axis=1)
        dist = (dist)%(2*torch.pi)

        dist[dist<-torch.pi] += 2*np.pi
        dist[dist>torch.pi] -= 2*np.pi
        
        return dist 
    # return torch.min(torch.abs(dist),2*torch.pi - torch.abs(dist))

def quat_log(q):
    v_norm = torch.linalg.norm(q[:,:3],axis=1)
    q_norm = torch.linalg.norm(q,axis=1)
    tolerance = 1e-17

    q_dist = torch.zeros((q.shape[0],4), dtype=torch.float, requires_grad=False).to(q.device)
    # print(q_dist)
    if torch.any(q_norm<tolerance):
        # 0 quaternion - undefined
        q_dist[q_norm<tolerance] = torch.tensor([torch.nan,torch.nan,torch.nan,-torch.inf]).to(q.device)
        raise "Undefined quaternion"
    # real quaternions - no imaginary part
    if torch.any(v_norm<tolerance):
        q_dist[v_norm<tolerance,0:3] = torch.tensor([0.,0.,0.]).to(q.device)
        q_dist[v_norm<tolerance,3] = torch.log(q_norm[v_norm<tolerance])
        
   
    rest_idx = (q_norm >= tolerance) * (v_norm >= tolerance)
    vec = q[rest_idx,:3] / (v_norm[rest_idx].unsqueeze(1))

    q_dist[rest_idx,:3] = torch.arccos(q[rest_idx,3]/q_norm[rest_idx]).unsqueeze(-1)*vec
    q_dist[rest_idx,3] = torch.log(q_norm[rest_idx])

    return q_dist


def quat_exp(q):

    tolerance = 1e-17
    v_norm = torch.linalg.norm(q[:,:3],axis=1)

    vec = q[:,:3]

    vec[v_norm > tolerance] = vec[v_norm>tolerance] / (v_norm[v_norm>tolerance].unsqueeze(1))
    magnitude = torch.exp(q[:,3])

    exp_q = torch.cat(((magnitude*torch.sin(v_norm)).unsqueeze(1)*vec, (magnitude*torch.cos(v_norm)).unsqueeze(1)),dim=1)
    return exp_q

def quat_slerp(q0,q1,steps):
    # From: https://splines.readthedocs.io/en/latest/rotation/slerp.html
    
    quat_angle = quat_distance(q1,q0)
    q1[quat_angle<0] *= -1 # Canonicalise the quaternion to ensure shortest distance is taken
    q0_conj = quat_conjugate(q0)
    q = quat_mul(q1,q0_conj)
    # q**t = exp(t⋅log(q))
    q_log = quat_log(q)
    q_steps = quat_exp(steps*q_log)
    q = quat_mul(q_steps,q0)
 
    return q

def torch_rand_float_tensor(lower, upper, shape, device):
    # Like torch_rang_float but accepts tensors as ranges. 
    # type: (Tensor[float], Tensor[float], Tuple[int, int], str) -> Tensor

    if lower.shape != shape or upper.shape != shape:
        raise ValueError("Lower and upper bounds must have the same shape as desired shape")
 
    return (upper - lower) * torch.rand(*shape, device=device) + lower

def points_in_nominal_pose_rectangle(points):
    rectangle = [0.2,-0.2, # x
                0.16,-0.16] # y

    x1,x2,y1,y2 = rectangle

    indexes = (points[:,:,0] < x1) * (points[:,:,0] > x2) * (points[:,:,1] < y1) * (points[:,:,1] > y2)

    points[indexes] = 1
    points[~indexes] = 0

    return points

# def torch_rand_float_seeded(lower, upper, shape, device):
#     # type: (float, float, Tuple[int, int], str) -> Tensor
#     return (upper - lower) * torch.rand(*shape, device=device) + lower

# def quat_conjugate(a):
#     shape = a.shape
#     return torch.cat((-a[:, :, :3], a[:, :, -1:]), dim=-1).view(shape)

# def quat_mul(a, b):
#     assert a.shape == b.shape
#     shape = a.shape

#     x1, y1, z1, w1 = a[:, :, 0], a[:, :, 1], a[:, :, 2], a[:, :, 3]
#     x2, y2, z2, w2 = b[:, :, 0], b[:, :,1], b[:, :,2], b[:, :, 3]
#     ww = (z1 + x1) * (x2 + y2)
#     yy = (w1 - y1) * (w2 + z2)
#     zz = (w1 + y1) * (w2 - z2)
#     xx = ww + yy + zz
#     qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
#     w = qq - ww + (z1 - y1) * (y2 - z2)
#     x = qq - xx + (x1 + w1) * (x2 + w2)
#     y = qq - yy + (w1 - x1) * (y2 + z2)
#     z = qq - zz + (z1 + y1) * (w2 - x2)

#     quat = torch.stack([x, y, z, w], dim=-1).view(shape)

#     return quat