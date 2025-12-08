import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import itertools
import tqdm
import copy
import scipy.stats as st
import os
import time
from collections import deque
from types import SimpleNamespace
from scipy.stats import norm
from sklearn.cluster import KMeans
from kneed import KneeLocator

# device = 'cpu'
device = 'cuda'

class LatentSegmentEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )

    def forward(self, x):
        return self.net(x)


def gen_net(in_size=1, out_size=1, H=128, n_layers=3, activation='tanh'):
    net = []
    for i in range(n_layers):
        net.append(nn.Linear(in_size, H))
        net.append(nn.LeakyReLU())
        in_size = H
    net.append(nn.Linear(in_size, out_size))
    if activation == 'tanh':
        net.append(nn.Tanh())
    elif activation == 'sig':
        net.append(nn.Sigmoid())
    else:
        net.append(nn.ReLU())

    return net

def KCenterGreedy(obs, full_obs, num_new_sample):
    selected_index = []
    current_index = list(range(obs.shape[0]))
    new_obs = obs
    new_full_obs = full_obs
    start_time = time.time()
    for count in range(num_new_sample):
        dist = compute_smallest_dist(new_obs, new_full_obs)
        max_index = torch.argmax(dist)
        max_index = max_index.item()
        
        if count == 0:
            selected_index.append(max_index)
        else:
            selected_index.append(current_index[max_index])
        current_index = current_index[0:max_index] + current_index[max_index+1:]
        
        new_obs = obs[current_index]
        new_full_obs = np.concatenate([
            full_obs, 
            obs[selected_index]], 
            axis=0)
    return selected_index

def compute_smallest_dist(obs, full_obs):
    obs = torch.from_numpy(obs).float()
    full_obs = torch.from_numpy(full_obs).float()
    batch_size = 100
    with torch.no_grad():
        total_dists = []
        for full_idx in range(len(obs) // batch_size + 1):
            full_start = full_idx * batch_size
            if full_start < len(obs):
                full_end = (full_idx + 1) * batch_size
                dists = []
                for idx in range(len(full_obs) // batch_size + 1):
                    start = idx * batch_size
                    if start < len(full_obs):
                        end = (idx + 1) * batch_size
                        dist = torch.norm(
                            obs[full_start:full_end, None, :].to(device) - full_obs[None, start:end, :].to(device), dim=-1, p=2
                        )
                        dists.append(dist)
                dists = torch.cat(dists, dim=1)
                small_dists = torch.torch.min(dists, dim=1).values
                total_dists.append(small_dists)
                
        total_dists = torch.cat(total_dists)
    return total_dists.unsqueeze(1)

class RewardModel:
    def __init__(self, ds, da, 
                 ensemble_size=3, lr=3e-4, mb_size = 128, size_segment=1, 
                 env_maker=None, max_size=100, activation='tanh', capacity=5e5,  
                 large_batch=1, label_margin=0.0, 
                 teacher_beta=-1, teacher_gamma=1, 
                 teacher_eps_mistake=0, 
                 teacher_eps_skip=0, 
                 teacher_eps_equal=0, 
                 cfg=None):
        
        # train data is trajectories, must process to sa and s..   
        self.ds = ds
        self.da = da
        self.de = ensemble_size
        self.lr = lr
        self.ensemble = []
        self.paramlst = []
        self.opt = None
        self.model = None
        self.max_size = max_size
        self.activation = activation
        self.size_segment = size_segment
        self.max_history_T = 3           # or 4–5 if you want longer flip windows
        self.pred_history = {} 
        self.capacity = int(capacity)
        self.buffer_seg1 = np.empty((self.capacity, size_segment, self.ds+self.da), dtype=np.float32)
        self.buffer_seg2 = np.empty((self.capacity, size_segment, self.ds+self.da), dtype=np.float32)
        self.buffer_label = np.empty((self.capacity, 1), dtype=np.float32)
        self.buffer_index = 0
        self.buffer_full = False
                
        self.construct_ensemble()
        self.inputs = []
        self.targets = []
        self.raw_actions = []
        self.img_inputs = []
        self.mb_size = mb_size
        self.origin_mb_size = mb_size
        self.train_batch_size = 128
        self.CEloss = nn.CrossEntropyLoss()
        self.running_means = []
        self.running_stds = []
        self.best_seg = []
        self.best_label = []
        self.best_action = []
        self.large_batch = large_batch
        
        # new teacher
        self.teacher_beta = teacher_beta
        self.teacher_gamma = teacher_gamma
        self.teacher_eps_mistake = teacher_eps_mistake
        self.teacher_eps_equal = teacher_eps_equal
        self.teacher_eps_skip = teacher_eps_skip
        self.teacher_thres_skip = 0
        self.teacher_thres_equal = 0
        
        self.label_margin = label_margin
        self.label_target = 1 - 2*self.label_margin

        if cfg is None:
            cfg = SimpleNamespace()
        self.cfg = cfg

    def softXEnt_loss(self, input, target):
        logprobs = torch.nn.functional.log_softmax (input, dim = 1)
        return  -(target * logprobs).sum() / input.shape[0]
    
    def change_batch(self, new_frac):
        self.mb_size = int(self.origin_mb_size*new_frac)
    
    def set_batch(self, new_batch):
        self.mb_size = int(new_batch)
        
    def set_teacher_thres_skip(self, new_margin):
        self.teacher_thres_skip = new_margin * self.teacher_eps_skip
        
    def set_teacher_thres_equal(self, new_margin):
        self.teacher_thres_equal = new_margin * self.teacher_eps_equal
        
    def construct_ensemble(self):
        for i in range(self.de):
            model = nn.Sequential(*gen_net(in_size=self.ds+self.da, 
                                           out_size=1, H=256, n_layers=3, 
                                           activation=self.activation)).float().to(device)
            self.ensemble.append(model)
            self.paramlst.extend(model.parameters())
            
        self.opt = torch.optim.Adam(self.paramlst, lr = self.lr)
            
    def add_data(self, obs, act, rew, done):
        sa_t = np.concatenate([obs, act], axis=-1)
        r_t = rew
        
        flat_input = sa_t.reshape(1, self.da+self.ds)
        r_t = np.array(r_t)
        flat_target = r_t.reshape(1, 1)

        init_data = len(self.inputs) == 0
        if init_data:
            self.inputs.append(flat_input)
            self.targets.append(flat_target)
        elif done:
            self.inputs[-1] = np.concatenate([self.inputs[-1], flat_input])
            self.targets[-1] = np.concatenate([self.targets[-1], flat_target])
            # FIFO
            if len(self.inputs) > self.max_size:
                self.inputs = self.inputs[1:]
                self.targets = self.targets[1:]
            self.inputs.append([])
            self.targets.append([])
        else:
            if len(self.inputs[-1]) == 0:
                self.inputs[-1] = flat_input
                self.targets[-1] = flat_target
            else:
                self.inputs[-1] = np.concatenate([self.inputs[-1], flat_input])
                self.targets[-1] = np.concatenate([self.targets[-1], flat_target])
                
    def add_data_batch(self, obses, rewards):
        num_env = obses.shape[0]
        for index in range(num_env):
            self.inputs.append(obses[index])
            self.targets.append(rewards[index])
        
    def get_rank_probability(self, x_1, x_2):
        # get probability x_1 > x_2
        probs = []
        for member in range(self.de):
            probs.append(self.p_hat_member(x_1, x_2, member=member).cpu().numpy())
        probs = np.array(probs)
        
        return np.mean(probs, axis=0), np.std(probs, axis=0)
    
    def get_entropy(self, x_1, x_2):
        # get probability x_1 > x_2
        probs = []
        for member in range(self.de):
            probs.append(self.p_hat_entropy(x_1, x_2, member=member).cpu().numpy())
        probs = np.array(probs)
        return np.mean(probs, axis=0), np.std(probs, axis=0)

    def p_hat_member(self, x_1, x_2, member=-1):
        # softmaxing to get the probabilities according to eqn 1
        with torch.no_grad():
            r_hat1 = self.r_hat_member(x_1, member=member)
            r_hat2 = self.r_hat_member(x_2, member=member)
            r_hat1 = r_hat1.sum(axis=1)
            r_hat2 = r_hat2.sum(axis=1)
            r_hat = torch.cat([r_hat1, r_hat2], axis=-1)
        
        # taking 0 index for probability x_1 > x_2
        return F.softmax(r_hat, dim=-1)[:,0]
    
    def p_hat_entropy(self, x_1, x_2, member=-1):
        # softmaxing to get the probabilities according to eqn 1
        with torch.no_grad():
            r_hat1 = self.r_hat_member(x_1, member=member)
            r_hat2 = self.r_hat_member(x_2, member=member)
            r_hat1 = r_hat1.sum(axis=1)
            r_hat2 = r_hat2.sum(axis=1)
            r_hat = torch.cat([r_hat1, r_hat2], axis=-1)
        
        ent = F.softmax(r_hat, dim=-1) * F.log_softmax(r_hat, dim=-1)
        ent = ent.sum(axis=-1).abs()
        return ent

    def r_hat_member(self, x, member=-1):
        # the network parameterizes r hat in eqn 1 from the paper
        return self.ensemble[member](torch.from_numpy(x).float().to(device))

    def r_hat(self, x):
        # they say they average the rewards from each member of the ensemble, but I think this only makes sense if the rewards are already normalized
        # but I don't understand how the normalization should be happening right now :(
        r_hats = []
        for member in range(self.de):
            r_hats.append(self.r_hat_member(x, member=member).detach().cpu().numpy())
        r_hats = np.array(r_hats)
        return np.mean(r_hats)
    
    def r_hat_batch(self, x):
        # they say they average the rewards from each member of the ensemble, but I think this only makes sense if the rewards are already normalized
        # but I don't understand how the normalization should be happening right now :(
        r_hats = []
        for member in range(self.de):
            r_hats.append(self.r_hat_member(x, member=member).detach().cpu().numpy())
        r_hats = np.array(r_hats)

        return np.mean(r_hats, axis=0)
    
    def save(self, model_dir, step):
        for member in range(self.de):
            torch.save(
                self.ensemble[member].state_dict(), '%s/reward_model_%s_%s.pt' % (model_dir, step, member)
            )
            
    def load(self, model_dir, step):
        for member in range(self.de):
            self.ensemble[member].load_state_dict(
                torch.load('%s/reward_model_%s_%s.pt' % (model_dir, step, member))
            )
    
    def get_train_acc(self):
        ensemble_acc = np.array([0 for _ in range(self.de)])
        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = np.random.permutation(max_len)
        batch_size = 256
        num_epochs = int(np.ceil(max_len/batch_size))
        
        total = 0
        for epoch in range(num_epochs):
            last_index = (epoch+1)*batch_size
            if (epoch+1)*batch_size > max_len:
                last_index = max_len
                
            sa_t_1 = self.buffer_seg1[epoch*batch_size:last_index]
            sa_t_2 = self.buffer_seg2[epoch*batch_size:last_index]
            labels = self.buffer_label[epoch*batch_size:last_index]
            labels = torch.from_numpy(labels.flatten()).long().to(device)
            total += labels.size(0)
            for member in range(self.de):
                # get logits
                r_hat1 = self.r_hat_member(sa_t_1, member=member)
                r_hat2 = self.r_hat_member(sa_t_2, member=member)
                r_hat1 = r_hat1.sum(axis=1)
                r_hat2 = r_hat2.sum(axis=1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)                
                _, predicted = torch.max(r_hat.data, 1)
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct
                
        ensemble_acc = ensemble_acc / total
        return np.mean(ensemble_acc)
    
    def get_queries(self, mb_size=20):
        len_traj, max_len = len(self.inputs[0]), len(self.inputs)
        img_t_1, img_t_2 = None, None
        
        if len(self.inputs[-1]) < len_traj:
            max_len = max_len - 1
        
        # get train traj
        train_inputs = np.array(self.inputs[:max_len])
        train_targets = np.array(self.targets[:max_len])
   
        batch_index_2 = np.random.choice(max_len, size=mb_size, replace=True)
        sa_t_2 = train_inputs[batch_index_2] # Batch x T x dim of s&a
        r_t_2 = train_targets[batch_index_2] # Batch x T x 1
        
        batch_index_1 = np.random.choice(max_len, size=mb_size, replace=True)
        sa_t_1 = train_inputs[batch_index_1] # Batch x T x dim of s&a
        r_t_1 = train_targets[batch_index_1] # Batch x T x 1
                
        sa_t_1 = sa_t_1.reshape(-1, sa_t_1.shape[-1]) # (Batch x T) x dim of s&a
        r_t_1 = r_t_1.reshape(-1, r_t_1.shape[-1]) # (Batch x T) x 1
        sa_t_2 = sa_t_2.reshape(-1, sa_t_2.shape[-1]) # (Batch x T) x dim of s&a
        r_t_2 = r_t_2.reshape(-1, r_t_2.shape[-1]) # (Batch x T) x 1

        # Generate time index 
        time_index = np.array([list(range(i*len_traj,
                                            i*len_traj+self.size_segment)) for i in range(mb_size)])
        time_index_2 = time_index + np.random.choice(len_traj-self.size_segment, size=mb_size, replace=True).reshape(-1,1)
        time_index_1 = time_index + np.random.choice(len_traj-self.size_segment, size=mb_size, replace=True).reshape(-1,1)
        
        sa_t_1 = np.take(sa_t_1, time_index_1, axis=0) # Batch x size_seg x dim of s&a
        r_t_1 = np.take(r_t_1, time_index_1, axis=0) # Batch x size_seg x 1
        sa_t_2 = np.take(sa_t_2, time_index_2, axis=0) # Batch x size_seg x dim of s&a
        r_t_2 = np.take(r_t_2, time_index_2, axis=0) # Batch x size_seg x 1
                
        return sa_t_1, sa_t_2, r_t_1, r_t_2

    def put_queries(self, sa_t_1, sa_t_2, labels):
        total_sample = sa_t_1.shape[0]
        next_index = self.buffer_index + total_sample
        if next_index >= self.capacity:
            self.buffer_full = True
            maximum_index = self.capacity - self.buffer_index
            np.copyto(self.buffer_seg1[self.buffer_index:self.capacity], sa_t_1[:maximum_index])
            np.copyto(self.buffer_seg2[self.buffer_index:self.capacity], sa_t_2[:maximum_index])
            np.copyto(self.buffer_label[self.buffer_index:self.capacity], labels[:maximum_index])

            remain = total_sample - (maximum_index)
            if remain > 0:
                np.copyto(self.buffer_seg1[0:remain], sa_t_1[maximum_index:])
                np.copyto(self.buffer_seg2[0:remain], sa_t_2[maximum_index:])
                np.copyto(self.buffer_label[0:remain], labels[maximum_index:])

            self.buffer_index = remain
        else:
            np.copyto(self.buffer_seg1[self.buffer_index:next_index], sa_t_1)
            np.copyto(self.buffer_seg2[self.buffer_index:next_index], sa_t_2)
            np.copyto(self.buffer_label[self.buffer_index:next_index], labels)
            self.buffer_index = next_index
            
    def get_label(self, sa_t_1, sa_t_2, r_t_1, r_t_2):
        sum_r_t_1 = np.sum(r_t_1, axis=1)
        sum_r_t_2 = np.sum(r_t_2, axis=1)
        
        # skip the query
        if self.teacher_thres_skip > 0: 
            max_r_t = np.maximum(sum_r_t_1, sum_r_t_2)
            max_index = (max_r_t > self.teacher_thres_skip).reshape(-1)
            if sum(max_index) == 0:
                return None, None, None, None, []

            sa_t_1 = sa_t_1[max_index]
            sa_t_2 = sa_t_2[max_index]
            r_t_1 = r_t_1[max_index]
            r_t_2 = r_t_2[max_index]
            sum_r_t_1 = np.sum(r_t_1, axis=1)
            sum_r_t_2 = np.sum(r_t_2, axis=1)
        
        # equally preferable
        margin_index = (np.abs(sum_r_t_1 - sum_r_t_2) < self.teacher_thres_equal).reshape(-1)
        
        # perfectly rational
        seg_size = r_t_1.shape[1]
        temp_r_t_1 = r_t_1.copy()
        temp_r_t_2 = r_t_2.copy()
        for index in range(seg_size-1):
            temp_r_t_1[:,:index+1] *= self.teacher_gamma
            temp_r_t_2[:,:index+1] *= self.teacher_gamma
        sum_r_t_1 = np.sum(temp_r_t_1, axis=1)
        sum_r_t_2 = np.sum(temp_r_t_2, axis=1)
            
        rational_labels = 1*(sum_r_t_1 < sum_r_t_2)
        if self.teacher_beta > 0: # Bradley-Terry rational model
            r_hat = torch.cat([torch.Tensor(sum_r_t_1), 
                               torch.Tensor(sum_r_t_2)], axis=-1)
            r_hat = r_hat*self.teacher_beta
            ent = F.softmax(r_hat, dim=-1)[:, 1]
            labels = torch.bernoulli(ent).int().numpy().reshape(-1, 1)
        else:
            labels = rational_labels
        
        # making a mistake
        len_labels = labels.shape[0]
        rand_num = np.random.rand(len_labels)
        noise_index = rand_num <= self.teacher_eps_mistake
        labels[noise_index] = 1 - labels[noise_index]
 
        # equally preferable
        labels[margin_index] = -1 
        
        return sa_t_1, sa_t_2, r_t_1, r_t_2, labels
    
    def kcenter_sampling(self):
        
        # get queries
        num_init = self.mb_size*self.large_batch
        sa_t_1, sa_t_2, r_t_1, r_t_2 =  self.get_queries(
            mb_size=num_init)
        
        # get final queries based on kmeans clustering
        temp_sa_t_1 = sa_t_1[:,:,:self.ds]
        temp_sa_t_2 = sa_t_2[:,:,:self.ds]
        temp_sa = np.concatenate([temp_sa_t_1.reshape(num_init, -1),  
                                  temp_sa_t_2.reshape(num_init, -1)], axis=1)
        
        max_len = self.capacity if self.buffer_full else self.buffer_index
        
        tot_sa_1 = self.buffer_seg1[:max_len, :, :self.ds]
        tot_sa_2 = self.buffer_seg2[:max_len, :, :self.ds]
        tot_sa = np.concatenate([tot_sa_1.reshape(max_len, -1),  
                                 tot_sa_2.reshape(max_len, -1)], axis=1)
        
        selected_index = KCenterGreedy(temp_sa, tot_sa, self.mb_size)

        r_t_1, sa_t_1 = r_t_1[selected_index], sa_t_1[selected_index]
        r_t_2, sa_t_2 = r_t_2[selected_index], sa_t_2[selected_index]
        
        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2)
        
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)
        
        return len(labels)
    
    def kcenter_disagree_sampling(self):
        
        num_init = self.mb_size*self.large_batch
        num_init_half = int(num_init*0.5)
        
        # get queries
        sa_t_1, sa_t_2, r_t_1, r_t_2 =  self.get_queries(
            mb_size=num_init)
        
        # get final queries based on uncertainty
        _, disagree = self.get_rank_probability(sa_t_1, sa_t_2)
        top_k_index = (-disagree).argsort()[:num_init_half]
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]
        
        # get final queries based on kmeans clustering
        temp_sa_t_1 = sa_t_1[:,:,:self.ds]
        temp_sa_t_2 = sa_t_2[:,:,:self.ds]
        
        temp_sa = np.concatenate([temp_sa_t_1.reshape(num_init_half, -1),  
                                  temp_sa_t_2.reshape(num_init_half, -1)], axis=1)
        
        max_len = self.capacity if self.buffer_full else self.buffer_index
        
        tot_sa_1 = self.buffer_seg1[:max_len, :, :self.ds]
        tot_sa_2 = self.buffer_seg2[:max_len, :, :self.ds]
        tot_sa = np.concatenate([tot_sa_1.reshape(max_len, -1),  
                                 tot_sa_2.reshape(max_len, -1)], axis=1)
        
        selected_index = KCenterGreedy(temp_sa, tot_sa, self.mb_size)
        
        r_t_1, sa_t_1 = r_t_1[selected_index], sa_t_1[selected_index]
        r_t_2, sa_t_2 = r_t_2[selected_index], sa_t_2[selected_index]

        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2)
        
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)
        
        return len(labels)
    
    def kcenter_entropy_sampling(self):
        
        num_init = self.mb_size*self.large_batch
        num_init_half = int(num_init*0.5)
        
        # get queries
        sa_t_1, sa_t_2, r_t_1, r_t_2 =  self.get_queries(
            mb_size=num_init)
        
        
        # get final queries based on uncertainty
        entropy, _ = self.get_entropy(sa_t_1, sa_t_2)
        top_k_index = (-entropy).argsort()[:num_init_half]
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]
        
        # get final queries based on kmeans clustering
        temp_sa_t_1 = sa_t_1[:,:,:self.ds]
        temp_sa_t_2 = sa_t_2[:,:,:self.ds]
        
        temp_sa = np.concatenate([temp_sa_t_1.reshape(num_init_half, -1),  
                                  temp_sa_t_2.reshape(num_init_half, -1)], axis=1)
        
        max_len = self.capacity if self.buffer_full else self.buffer_index
        
        tot_sa_1 = self.buffer_seg1[:max_len, :, :self.ds]
        tot_sa_2 = self.buffer_seg2[:max_len, :, :self.ds]
        tot_sa = np.concatenate([tot_sa_1.reshape(max_len, -1),  
                                 tot_sa_2.reshape(max_len, -1)], axis=1)
        
        selected_index = KCenterGreedy(temp_sa, tot_sa, self.mb_size)
        
        r_t_1, sa_t_1 = r_t_1[selected_index], sa_t_1[selected_index]
        r_t_2, sa_t_2 = r_t_2[selected_index], sa_t_2[selected_index]

        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2)
        
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)
        
        return len(labels)
    
    def uniform_sampling(self):
        # get queries
        sa_t_1, sa_t_2, r_t_1, r_t_2 =  self.get_queries(
            mb_size=self.mb_size)
            
        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2)
        
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)
        
        return len(labels)
    
    def disagreement_sampling(self):
        
        # get queries
        sa_t_1, sa_t_2, r_t_1, r_t_2 =  self.get_queries(
            mb_size=self.mb_size*self.large_batch)
        
        # get final queries based on uncertainty
        _, disagree = self.get_rank_probability(sa_t_1, sa_t_2)
        top_k_index = (-disagree).argsort()[:self.mb_size]
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]        
        
        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2)        
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)
        
        return len(labels)
    
    def entropy_sampling(self):
        
        # get queries
        sa_t_1, sa_t_2, r_t_1, r_t_2 =  self.get_queries(
            mb_size=self.mb_size*self.large_batch)
        
        # get final queries based on uncertainty
        entropy, _ = self.get_entropy(sa_t_1, sa_t_2)
        
        top_k_index = (-entropy).argsort()[:self.mb_size]
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]
        
        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(    
            sa_t_1, sa_t_2, r_t_1, r_t_2)
        
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)
        
        return len(labels)
    
    def compute_on_policy_measure(self, trajectories, policy_probs):
        """
        Compute on-policy measure O(τ) for trajectories.
        
        Args:
            trajectories: list of trajectory data from replay buffer
            policy_probs: log probabilities of actions under current policy for each trajectory
        
        Returns:
            on_policy_scores: rectified Z-scores for each trajectory
        """
        # Compute sum of log probabilities for each trajectory
        on_policy_raw = np.array([np.sum(log_probs) for log_probs in policy_probs])
        
        # Compute mean and std
        mu = np.mean(on_policy_raw)
        sigma = np.std(on_policy_raw)
        
        # Compute rectified Z-scores
        if sigma > 0:
            z_scores = (on_policy_raw - mu) / sigma
            on_policy_scores = np.maximum(0, z_scores)
        else:
            on_policy_scores = np.zeros_like(on_policy_raw)
        
        return on_policy_scores

    def _hash_pair(self, seg1, seg2):
        # seg1, seg2: (size_segment, ds+da) numpy arrays
        # Hash to a stable key; you can also use murmurhash or sha1 if you prefer.
        return hash(seg1.tobytes() + b"|" + seg2.tobytes())

    def _update_prob_history(self, sa_t_1, sa_t_2, probs_now):
        """
        sa_t_*: (N, size_segment, ds+da)
        probs_now: (M, N) current ensemble probs for these N queries
        """
        M, N = probs_now.shape
        for j in range(N):
            key = self._hash_pair(sa_t_1[j], sa_t_2[j])
            dq = self.pred_history.get(key)
            if dq is None:
                dq = deque(maxlen=self.max_history_T)
                self.pred_history[key] = dq
            dq.append(probs_now[:, j].copy())  # store (M,) for this query at this time

    def _collect_probs_history(self, sa_t_1, sa_t_2, require_full_window=False):
        """
        Reassemble per-time (M, N) arrays for the current queries from the per-query deques.
        Returns: list of length T with each element (M, N), or None if insufficient history.
        """
        # Find the minimal T that all queries have available
        N = sa_t_1.shape[0]
        per_query_hist = []
        lengths = []
        for j in range(N):
            key = self._hash_pair(sa_t_1[j], sa_t_2[j])
            dq = self.pred_history.get(key)
            if dq is None or len(dq) < 2:   # need at least 2 checkpoints to count flips
                return None
            per_query_hist.append(dq)
            lengths.append(len(dq))

        # If you want a strict window length = self.max_history_T:
        if require_full_window and min(lengths) < self.max_history_T:
            return None

        T = min(lengths)  # align all queries to same shortest history
        # Build list of (M, N): time-major
        out = []
        for t in range(T):
            # collect per-query (M,) at position -T + t
            cols = [per_query_hist[j][len(per_query_hist[j]) - T + t] for j in range(N)]
            # stack to (N, M) then transpose to (M, N)
            mat = np.stack(cols, axis=0).T
            out.append(mat)
        return out  # length T, each (M, N)

    def init_latent_training(self, lr=1e-4, temperature=0.1):
        """
        Enable online training of the latent encoder with InfoNCE.
        Call this AFTER load_latent_encoder().
        """
        if getattr(self, "latent_encoder", None) is None:
            print("[LatentEncoder] init_latent_training called but latent_encoder is None")
            return

        self.temperature = temperature

        # Make sure encoder is trainable
        self.latent_encoder.train()
        for p in self.latent_encoder.parameters():
            p.requires_grad = True

        self.latent_optimizer = torch.optim.Adam(
            self.latent_encoder.parameters(), lr=lr
        )

        print("[LatentEncoder] Online training enabled")
        print(f"[LatentEncoder] LR={lr}, Temp={temperature}")



    def latent_infonce_loss(self, z_anchor, z_pos, z_neg):
        """
        z_anchor: (B, d)
        z_pos:    (B, d)
        z_neg:    (B, d) or (B, K, d)
        """

        # Cosine similarities
        sim_pos = torch.nn.functional.cosine_similarity(z_anchor, z_pos)

        # If negative is shape (B, d), expand to (B, 1, d)
        if z_neg.dim() == 2:
            z_neg = z_neg.unsqueeze(1)

        sim_neg = torch.nn.functional.cosine_similarity(
            z_anchor.unsqueeze(1),  # (B,1,d)
            z_neg,                  # (B,K,d)
            dim=-1
        )

        # Build logits
        logits = torch.cat([sim_pos.unsqueeze(1), sim_neg], dim=1)
        logits = logits / self.temperature

        # Positive is index 0
        labels = torch.zeros(
            logits.size(0), dtype=torch.long, device=logits.device
        )

        loss = torch.nn.functional.cross_entropy(logits, labels)

        print(f"[LatentEncoder] InfoNCE loss={loss.item():.4f}")

        return loss

    def train_latent_encoder(self, z_anchor, z_pos, z_neg):
        loss = self.latent_infonce_loss(z_anchor, z_pos, z_neg)

        self.latent_optimizer.zero_grad()
        loss.backward()
        self.latent_optimizer.step()

        print("[LatentEncoder] Updated encoder weights")

        return loss.item()



    def load_latent_encoder(self, ckpt_path, device):
        """
        Loading a pre-trained segment encoder from an autoencoder checkpoint.
        Handles both:
          - state_dict with keys "0.weight", "2.bias", ...
          - state_dict with keys "net.0.weight", ...
        """
        if not os.path.exists(ckpt_path):
            print(f"[LatentEncoder] No checkpoint found at {ckpt_path}. Skipping.")
            self.latent_encoder = None
            return

        ckpt = torch.load(ckpt_path, map_location=device)

        input_dim   = ckpt["input_dim"]
        latent_dim  = ckpt["latent_dim"]
        seg_len     = ckpt["segment_len"]
        feature_dim = ckpt["feature_dim"]

        if hasattr(self, "size_segment") and seg_len != self.size_segment:
            print(f"[LatentEncoder][Warning] segment_len mismatch: "
                  f"ckpt={seg_len}, reward_model={self.size_segment}")
        if hasattr(self, "ds") and hasattr(self, "da") and feature_dim != (self.ds + self.da):
            print(f"[LatentEncoder][Warning] feature_dim mismatch: "
                  f"ckpt={feature_dim}, reward_model={self.ds + self.da}")

        encoder = LatentSegmentEncoder(input_dim, latent_dim)

        # --- Fix key mismatch between checkpoint and LatentSegmentEncoder ---
        raw_state = ckpt["encoder_state_dict"]

        # If the keys look like "0.weight" instead of "net.0.weight", add the prefix.
        if not any(k.startswith("net.") for k in raw_state.keys()):
            print("[LatentEncoder] Detected flat keys in state_dict, "
                  "adding 'net.' prefix to match LatentSegmentEncoder.")
            fixed_state = {}
            for k, v in raw_state.items():
                fixed_state[f"net.{k}"] = v
            raw_state = fixed_state

        try:
            encoder.load_state_dict(raw_state, strict=True)
        except RuntimeError as e:
            print(f"[LatentEncoder] Still failed to load state_dict: {e}")
            print("[LatentEncoder] Disabling latent encoder and falling back to baseline DUO.")
            self.latent_encoder = None
            return

        encoder.to(device)
        encoder.train()   # we want it trainable online

        self.latent_encoder   = encoder
        self.latent_input_dim = input_dim
        self.latent_dim       = latent_dim
        self.latent_seg_len   = seg_len
        self.latent_feat_dim  = feature_dim
        self.latent_device    = device

        print(f"[LatentEncoder] Loaded encoder from {ckpt_path}")
        print(f"[LatentEncoder] input_dim={input_dim}, latent_dim={latent_dim}, "
              f"seg_len={seg_len}, feat_dim={feature_dim}")

        # Kick off online training setup
        lr  = getattr(self.cfg, "latent_lr", 1e-3)
        tau = getattr(self.cfg, "latent_temperature", 0.1)
        self.init_latent_training(lr=lr, temperature=tau)





    def encode_segments_latent(self, segments, grad: bool = False):
        """
        segments: (N, T, ds+da) as np.ndarray or torch.Tensor

        grad=False: use no_grad() (for DUO Stage 3 clustering).
        grad=True : keep computation graph (for InfoNCE training).
        """
        if getattr(self, "latent_encoder", None) is None:
            raise RuntimeError(
                "encode_segments_latent called but latent_encoder is None. "
                "Did you call load_latent_encoder()?"
            )

        if isinstance(segments, np.ndarray):
            x = torch.from_numpy(segments.astype(np.float32))
        else:
            x = segments.float()

        N, T, D = x.shape
        x = x.view(N, T * D).to(self.latent_device)

        if grad:
            z = self.latent_encoder(x)          # keep graph for backprop
        else:
            with torch.no_grad():
                z = self.latent_encoder(x)      # just features

        return z





    def _gather_ensemble_probs(self, sa_t_1, sa_t_2):
        """
        Returns: np.array shape (M, N) where M = self.de and N = num_queries
        """
        probs = []
        for member in range(self.de):
            prob = self.p_hat_member(sa_t_1, sa_t_2, member=member).cpu().numpy()
            probs.append(prob)
        return np.array(probs)

    def get_epistemic_uncertainty_entropy(self, sa_t_1, sa_t_2):
        eps = 1e-12
        probs = []
        for member in range(self.de):
            prob = self.p_hat_member(sa_t_1, sa_t_2, member=member).cpu().numpy()
            probs.append(prob)
        
        probs = np.array(probs) 
        all_prefer_1 = np.all(probs > 0.5, axis=0)
        all_prefer_2 = np.all(probs < 0.5, axis=0)
        consensus_mask = ~(all_prefer_1 | all_prefer_2)

        mu = probs.mean(axis=0)
        mu = np.clip(mu, eps, 1.0 - eps)
        uncertainties = - (mu * np.log(mu) + (1.0 - mu) * np.log(1.0 - mu))
        return uncertainties, consensus_mask

    def get_epistemic_uncertainty_bald(self, sa_t_1, sa_t_2):
        eps = 1e-12
        probs = []
        for member in range(self.de):
            prob = self.p_hat_member(sa_t_1, sa_t_2, member=member).cpu().numpy()
            probs.append(prob)
        
        probs = np.array(probs) 
        all_prefer_1 = np.all(probs > 0.5, axis=0)
        all_prefer_2 = np.all(probs < 0.5, axis=0)
        consensus_mask = ~(all_prefer_1 | all_prefer_2)

        M = float(probs.shape[0])
        mu = probs.mean(axis=0)
        mu = np.clip(mu, eps, 1.0 - eps)
        H_mu = - (mu * np.log(mu) + (1.0 - mu) * np.log(1.0 - mu))

        # per-model entropies, then mean
        per_model_H = - (probs * np.log(np.clip(probs, eps, 1.0)) + (1.0 - probs) * np.log(np.clip(1.0 - probs, eps, 1.0)))
        H_each = per_model_H.mean(axis=0)

        uncertainties = H_mu - H_each
        return uncertainties, consensus_mask

    def get_epistemic_uncertainty_margin(self, sa_t_1, sa_t_2):
        probs = []
        for member in range(self.de):
            prob = self.p_hat_member(sa_t_1, sa_t_2, member=member).cpu().numpy()
            probs.append(prob)
        
        probs = np.array(probs) 
        all_prefer_1 = np.all(probs > 0.5, axis=0)
        all_prefer_2 = np.all(probs < 0.5, axis=0)
        consensus_mask = ~(all_prefer_1 | all_prefer_2)

        mu = probs.mean(axis=0)
        uncertainties = 1.0 - np.abs(0.5 - mu)  #higher -> more uncertain
        return uncertainties, consensus_mask

    def get_epistemic_uncertainty_variance(self, sa_t_1, sa_t_2):
        probs = []
        for member in range(self.de):
            prob = self.p_hat_member(sa_t_1, sa_t_2, member=member).cpu().numpy()
            probs.append(prob)
        
        probs = np.array(probs) 
        all_prefer_1 = np.all(probs > 0.5, axis=0)
        all_prefer_2 = np.all(probs < 0.5, axis=0)
        consensus_mask = ~(all_prefer_1 | all_prefer_2)

        uncertainties = probs.var(axis=0)
        return uncertainties, consensus_mask

    def get_epistemic_uncertainty_flip(self, sa_t_1, sa_t_2, probs_history):
        """
        probs_history: list of (M,N) across checkpoints, oldest->newest
        Returns: (uncertainties, consensus_mask)
        """
        assert isinstance(probs_history, (list, tuple)) and len(probs_history) >= 2
        eps = 1e-12
        votes_hist = []
        for P in probs_history:
            P = np.clip(np.asarray(P), eps, 1.0 - eps)
            votes_hist.append((P > 0.5).mean(axis=0) > 0.5)
        votes_hist = np.stack(votes_hist, axis=0)  # (T,N)
        flips = (votes_hist[1:] != votes_hist[:-1]).sum(axis=0).astype(float)
        uncertainties = flips / (votes_hist.shape[0] - 1 + eps)

        probs_now = np.asarray(probs_history[-1])
        all_prefer_1 = np.all(probs_now > 0.5, axis=0)
        all_prefer_2 = np.all(probs_now < 0.5, axis=0)
        consensus_mask = ~(all_prefer_1 | all_prefer_2)
        return uncertainties, consensus_mask

    def get_epistemic_uncertainty_perturb(self, sa_t_1, sa_t_2, num_perturb=3, noise_std=0.01, clip_range=None):
        eps = 1e-12
        base_probs = self._gather_ensemble_probs(sa_t_1, sa_t_2).mean(axis=0)  # (N,)
        deltas = []
        for _ in range(num_perturb):
            n1 = np.random.normal(0.0, noise_std, size=sa_t_1.shape).astype(sa_t_1.dtype)
            n2 = np.random.normal(0.0, noise_std, size=sa_t_2.shape).astype(sa_t_2.dtype)
            p1, p2 = sa_t_1 + n1, sa_t_2 + n2
            if clip_range is not None:
                lo, hi = clip_range
                p1 = np.clip(p1, lo, hi); p2 = np.clip(p2, lo, hi)
            pert_probs = self._gather_ensemble_probs(p1, p2).mean(axis=0)
            deltas.append(np.abs(pert_probs - base_probs))
        uncertainties = np.mean(np.stack(deltas, axis=0), axis=0)

        probs_now = self._gather_ensemble_probs(sa_t_1, sa_t_2)
        all_prefer_1 = np.all(probs_now > 0.5, axis=0)
        all_prefer_2 = np.all(probs_now < 0.5, axis=0)
        consensus_mask = ~(all_prefer_1 | all_prefer_2)
        return uncertainties, consensus_mask

    def get_epistemic_uncertainty_query_alignment(self, sa_t_1, sa_t_2, on_policy_weights=None):
        eps = 1e-12
        probs = self._gather_ensemble_probs(sa_t_1, sa_t_2)
        mu = np.clip(probs.mean(axis=0), eps, 1.0 - eps)
        ent = - (mu*np.log(mu) + (1-mu)*np.log(1-mu))
        ent_norm = ent / (np.log(2)+eps)
        var = probs.var(axis=0)
        if on_policy_weights is None:
            on_policy_weights = np.ones_like(mu)
        score = on_policy_weights * var * (1.0 - ent_norm)

        all_prefer_1 = np.all(probs > 0.5, axis=0)
        all_prefer_2 = np.all(probs < 0.5, axis=0)
        consensus_mask = ~(all_prefer_1 | all_prefer_2)
        return score, consensus_mask

    def get_epistemic_uncertainty_annotator_ease(self, sa_t_1, sa_t_2, entropy_cap=0.3):
        eps = 1e-12
        probs = self._gather_ensemble_probs(sa_t_1, sa_t_2)
        mu = np.clip(probs.mean(axis=0), eps, 1.0 - eps)
        ent = - (mu*np.log(mu) + (1-mu)*np.log(1-mu))
        ent_norm = ent / (np.log(2)+eps)
        var = probs.var(axis=0)
        score = var * (1.0 - ent_norm)
        if entropy_cap is not None:
            score = np.where(ent_norm <= entropy_cap, score, 0.0)

        all_prefer_1 = np.all(probs > 0.5, axis=0)
        all_prefer_2 = np.all(probs < 0.5, axis=0)
        consensus_mask = ~(all_prefer_1 | all_prefer_2)
        return score, consensus_mask





    def get_epistemic_uncertainty(self, sa_t_1, sa_t_2):
        """
        Compute epistemic uncertainty as the length of predicted preference interval.
        
        Args:
            sa_t_1: first segment
            sa_t_2: second segment
        
        Returns:
            uncertainties: preference interval length for each query
            consensus_mask: mask indicating non-consensual queries
        """
        # Get predictions from all ensemble members
        probs = []
        for member in range(self.de):
            prob = self.p_hat_member(sa_t_1, sa_t_2, member=member).cpu().numpy()
            probs.append(prob)
        
        probs = np.array(probs)  # Shape: (ensemble_size, num_queries)
        
        # Filter consensual predictions
        # Consensual: all members agree (all > 0.5 or all < 0.5)
        all_prefer_1 = np.all(probs > 0.5, axis=0)
        all_prefer_2 = np.all(probs < 0.5, axis=0)
        consensus_mask = ~(all_prefer_1 | all_prefer_2)
        
        # Compute epistemic uncertainty as preference interval length
        uncertainties = np.max(probs, axis=0) - np.min(probs, axis=0)
        
        return uncertainties, consensus_mask

    def get_reward_differences(self, sa_t_1, sa_t_2):
        """
        Compute predicted reward difference sequences for query pairs.
        
        Args:
            sa_t_1: first segment (batch_size, segment_length, state_action_dim)
            sa_t_2: second segment (batch_size, segment_length, state_action_dim)
        
        Returns:
            reward_diffs: reward difference vectors (batch_size, 2*segment_length)
        """
        batch_size = sa_t_1.shape[0]
        segment_length = sa_t_1.shape[1]
        
        # Get predicted rewards for both segments (averaged over ensemble)
        r_hat_1 = self.r_hat_batch(sa_t_1.reshape(-1, sa_t_1.shape[-1]))
        r_hat_2 = self.r_hat_batch(sa_t_2.reshape(-1, sa_t_2.shape[-1]))
        
        # Reshape to (batch_size, segment_length)
        r_hat_1 = r_hat_1.reshape(batch_size, segment_length)
        r_hat_2 = r_hat_2.reshape(batch_size, segment_length)
        
        # Compute reward differences and concatenate
        reward_diffs = np.concatenate([r_hat_1, r_hat_2], axis=1)
        
        return reward_diffs
    
    def adaptive_kmeans_clustering(self, features, max_k=None):
        """
        Apply K-means clustering with adaptive K using elbow method.
        
        Args:
            features: feature vectors to cluster
            max_k: maximum number of clusters to consider
        
        Returns:
            cluster_centers: indices of samples closest to cluster centers
            n_clusters: number of clusters selected
        """
        n_samples = features.shape[0]
        
        if max_k is None:
            max_k = min(n_samples // 2, 20)  # Reasonable upper bound
        
        max_k = max(2, min(max_k, n_samples))
        
        # If we have very few samples, just return all
        if n_samples <= 2:
            return list(range(n_samples)), n_samples
        
        # Compute inertias for different k values
        inertias = []
        k_range = range(2, max_k + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(features)
            inertias.append(kmeans.inertia_)
        
        # Use elbow method to find optimal k
        try:
            kl = KneeLocator(list(k_range), inertias, curve='convex', direction='decreasing')
            optimal_k = kl.elbow if kl.elbow is not None else max_k // 2
        except:
            # Fallback: use middle value
            optimal_k = max_k // 2
        
        # Perform final clustering with optimal k
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        kmeans.fit(features)
        
        # Find samples closest to cluster centers
        cluster_centers_idx = []
        for i in range(optimal_k):
            cluster_mask = kmeans.labels_ == i
            cluster_points = features[cluster_mask]
            cluster_indices = np.where(cluster_mask)[0]
            
            # Find point closest to center
            distances = np.linalg.norm(cluster_points - kmeans.cluster_centers_[i], axis=1)
            closest_idx = cluster_indices[np.argmin(distances)]
            cluster_centers_idx.append(closest_idx)
        
        return cluster_centers_idx, optimal_k
    
    # def duo_sampling(self, policy_log_probs=None):
    #     print("Running Duo sampling...")
    #     num_init = self.mb_size * self.large_batch
        
    #     # Stage 1: On-policy query generation (ξO)
    #     if policy_log_probs is not None and len(policy_log_probs) > 0:
    #         # Compute on-policy scores
    #         on_policy_scores = self.compute_on_policy_measure(self.inputs, policy_log_probs)
            
    #         # Normalize to probabilities
    #         if np.sum(on_policy_scores) > 0:
    #             sampling_probs = on_policy_scores / np.sum(on_policy_scores)
    #         else:
    #             sampling_probs = None
    #     else:
    #         sampling_probs = None
        
    #     # Generate queries with on-policy bias
    #     len_traj, max_len = len(self.inputs[0]), len(self.inputs)
    #     if len(self.inputs[-1]) < len_traj:
    #         max_len = max_len - 1
        
    #     train_inputs = np.array(self.inputs[:max_len])
    #     train_targets = np.array(self.targets[:max_len])
        
    #     # Sample trajectories with on-policy bias
    #     if sampling_probs is not None:
    #         batch_index_1 = np.random.choice(max_len, size=num_init, replace=True, p=sampling_probs[:max_len])
    #         batch_index_2 = np.random.choice(max_len, size=num_init, replace=True, p=sampling_probs[:max_len])
    #     else:
    #         batch_index_1 = np.random.choice(max_len, size=num_init, replace=True)
    #         batch_index_2 = np.random.choice(max_len, size=num_init, replace=True)
        
    #     sa_t_1 = train_inputs[batch_index_1]
    #     sa_t_2 = train_inputs[batch_index_2]
    #     r_t_1 = train_targets[batch_index_1]
    #     r_t_2 = train_targets[batch_index_2]
        
    #     # Generate segment indices
    #     sa_t_1 = sa_t_1.reshape(-1, sa_t_1.shape[-1])
    #     r_t_1 = r_t_1.reshape(-1, r_t_1.shape[-1])
    #     sa_t_2 = sa_t_2.reshape(-1, sa_t_2.shape[-1])
    #     r_t_2 = r_t_2.reshape(-1, r_t_2.shape[-1])
        
    #     time_index = np.array([list(range(i*len_traj, i*len_traj+self.size_segment)) for i in range(num_init)])
    #     time_index_1 = time_index + np.random.choice(len_traj-self.size_segment, size=num_init, replace=True).reshape(-1,1)
    #     time_index_2 = time_index + np.random.choice(len_traj-self.size_segment, size=num_init, replace=True).reshape(-1,1)
        
    #     sa_t_1 = np.take(sa_t_1, time_index_1, axis=0)
    #     r_t_1 = np.take(r_t_1, time_index_1, axis=0)
    #     sa_t_2 = np.take(sa_t_2, time_index_2, axis=0)
    #     r_t_2 = np.take(r_t_2, time_index_2, axis=0)


    #     # --- Per-query on-policy weights for current candidate pairs ---
    #     on_policy_weights = None
    #     if policy_log_probs is not None and len(policy_log_probs) > 0:
    #         # We assume: policy_log_probs[t] is a 1D array/list of per-step log-probs for trajectory t,
    #         # and every trajectory has length len_traj (same as above).
    #         num_init = sa_t_1.shape[0]
    #         starts_1 = (time_index_1[:, 0] - np.arange(num_init) * len_traj)  # start idx within traj
    #         starts_2 = (time_index_2[:, 0] - np.arange(num_init) * len_traj)

    #         # Sum log-probs over each selected segment window
    #         seg_logsum_1 = np.array([
    #             np.sum(policy_log_probs[batch_index_1[i]][starts_1[i] : starts_1[i] + self.size_segment])
    #             for i in range(num_init)
    #         ], dtype=float)
    #         seg_logsum_2 = np.array([
    #             np.sum(policy_log_probs[batch_index_2[i]][starts_2[i] : starts_2[i] + self.size_segment])
    #             for i in range(num_init)
    #         ], dtype=float)

    #         # Rectified z-scores (segment-level, like your compute_on_policy_measure but for windows)
    #         all_seg = np.concatenate([seg_logsum_1, seg_logsum_2])
    #         mu = all_seg.mean()
    #         sigma = all_seg.std()
    #         if sigma > 0:
    #             z1 = np.maximum(0.0, (seg_logsum_1 - mu) / sigma)
    #             z2 = np.maximum(0.0, (seg_logsum_2 - mu) / sigma)
    #         else:
    #             z1 = np.zeros_like(seg_logsum_1)
    #             z2 = np.zeros_like(seg_logsum_2)

    #         # Pair weight: geometric mean emphasizes “both segments are on-policy”
    #         on_policy_weights = np.sqrt(z1 * z2)

    #         # Normalize to [0,1] for stability; fall back to ones if all zero
    #         maxw = on_policy_weights.max() if on_policy_weights.size > 0 else 0.0
    #         if maxw > 0:
    #             on_policy_weights = on_policy_weights / maxw
    #         else:
    #             on_policy_weights = np.ones_like(on_policy_weights)
    #     else:
    #         # No policy logs → just use uniform weights
    #         on_policy_weights = np.ones(sa_t_1.shape[0], dtype=float)


    #     # probs_now = self._gather_ensemble_probs(sa_t_1, sa_t_2)
    #     # self._update_prob_history(sa_t_1, sa_t_2, probs_now)
    #     # probs_history = self._collect_probs_history(sa_t_1, sa_t_2, require_full_window=False)
    #     # if probs_history is not None:
    #     #     uncertainties, consensus_mask = self.get_epistemic_uncertainty_flip(sa_t_1, sa_t_2, probs_history)
    #     # else:
    #     #     uncertainties, consensus_mask = self.get_epistemic_uncertainty_variance(sa_t_1, sa_t_2)

    #     # Stage 2: Uncertain query selection (ξU)
    #     uncertainties, consensus_mask = self.get_epistemic_uncertainty(sa_t_1, sa_t_2)
    #     # uncertainties, consensus_mask = self.get_epistemic_uncertainty_variance(sa_t_1, sa_t_2)
    #     # uncertainties, consensus_mask = self.get_epistemic_uncertainty_perturb(sa_t_1, sa_t_2, num_perturb=3, noise_std=0.01)
    #     # optional: on_policy_weights = <np.ndarray shape (N,)>
    #     # uncertainties, consensus_mask = self.get_epistemic_uncertainty_query_alignment(sa_t_1, sa_t_2, on_policy_weights=on_policy_weights)  # or ... , on_policy_weights=on_policy_weights
    #     # uncertainties, consensus_mask = self.get_epistemic_uncertainty_annotator_ease(sa_t_1, sa_t_2, entropy_cap=0.3)

        
    #     # Filter out consensual queries
    #     if np.sum(consensus_mask) == 0:
    #         # All queries are consensual, fall back to top uncertain ones
    #         num_uncertain = min(self.mb_size * 2, num_init)
    #         top_uncertain_idx = (-uncertainties).argsort()[:num_uncertain]
    #     else:
    #         # Keep only non-consensual queries and sort by uncertainty
    #         non_consensual_idx = np.where(consensus_mask)[0]
    #         non_consensual_uncertainties = uncertainties[consensus_mask]
            
    #         # Sort by uncertainty and keep top candidates
    #         num_uncertain = min(self.mb_size * 2, len(non_consensual_idx))
    #         sorted_idx = (-non_consensual_uncertainties).argsort()[:num_uncertain]
    #         top_uncertain_idx = non_consensual_idx[sorted_idx]
        
    #     sa_t_1 = sa_t_1[top_uncertain_idx]
    #     sa_t_2 = sa_t_2[top_uncertain_idx]
    #     r_t_1 = r_t_1[top_uncertain_idx]
    #     r_t_2 = r_t_2[top_uncertain_idx]
        
    #     # Stage 3: Diverse query selection (ξD)
    #     # Compute reward difference features
    #     # reward_diffs = self.get_reward_differences(sa_t_1, sa_t_2)
    #     mode = getattr(self.cfg, "duo_stage3", "baseline")  # default to baseline

    #     if mode == "latent" and getattr(self, "latent_encoder", None) is not None:
    #         z1 = self.encode_segments_latent(sa_t_1)
    #         z2 = self.encode_segments_latent(sa_t_2)
    #         print("In latent space!!")
    #         u = torch.cat([z1, z2, z1 - z2], dim=-1)
    #         features = u.detach().cpu().numpy()
    #     else:
    #         print("baseline mode!")
    #         reward_diffs = self.get_reward_differences(sa_t_1, sa_t_2)
    #         features = reward_diffs
        
    #     # Apply adaptive K-means clustering on chosen feature space
    #     num_diverse = min(self.mb_size, len(features))
    #     if len(features) <= num_diverse:
    #         # Not enough queries, use all
    #         selected_index = list(range(len(features)))
    #     else:
    #         selected_index, _ = self.adaptive_kmeans_clustering(features, max_k=num_diverse)

        
    #     sa_t_1 = sa_t_1[selected_index]
    #     sa_t_2 = sa_t_2[selected_index]
    #     r_t_1 = r_t_1[selected_index]
    #     r_t_2 = r_t_2[selected_index]
        
    #     # Get labels
    #     sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(
    #         sa_t_1, sa_t_2, r_t_1, r_t_2)
        
    #     if len(labels) > 0:
    #         self.put_queries(sa_t_1, sa_t_2, labels)
        
    #     return len(labels)

    def duo_sampling(self, policy_log_probs=None):
        print("Running Duo sampling...")
        num_init = self.mb_size * self.large_batch

        # ----------------- Stage 1: On-policy query generation (ξO) -----------------
        if policy_log_probs is not None and len(policy_log_probs) > 0:
            on_policy_scores = self.compute_on_policy_measure(self.inputs, policy_log_probs)
            if np.sum(on_policy_scores) > 0:
                sampling_probs = on_policy_scores / np.sum(on_policy_scores)
            else:
                sampling_probs = None
        else:
            sampling_probs = None

        len_traj, max_len = len(self.inputs[0]), len(self.inputs)
        if len(self.inputs[-1]) < len_traj:
            max_len = max_len - 1

        train_inputs = np.array(self.inputs[:max_len])
        train_targets = np.array(self.targets[:max_len])

        # sample trajectories (with or without on-policy bias)
        if sampling_probs is not None:
            batch_index_1 = np.random.choice(max_len, size=num_init, replace=True,
                                             p=sampling_probs[:max_len])
            batch_index_2 = np.random.choice(max_len, size=num_init, replace=True,
                                             p=sampling_probs[:max_len])
        else:
            batch_index_1 = np.random.choice(max_len, size=num_init, replace=True)
            batch_index_2 = np.random.choice(max_len, size=num_init, replace=True)

        sa_t_1 = train_inputs[batch_index_1]
        sa_t_2 = train_inputs[batch_index_2]
        r_t_1 = train_targets[batch_index_1]
        r_t_2 = train_targets[batch_index_2]

        # flatten over time so we can pick sub-segments
        sa_t_1 = sa_t_1.reshape(-1, sa_t_1.shape[-1])
        r_t_1 = r_t_1.reshape(-1, r_t_1.shape[-1])
        sa_t_2 = sa_t_2.reshape(-1, sa_t_2.shape[-1])
        r_t_2 = r_t_2.reshape(-1, r_t_2.shape[-1])

        time_index = np.array([
            list(range(i * len_traj, i * len_traj + self.size_segment))
            for i in range(num_init)
        ])
        time_index_1 = time_index + np.random.choice(
            len_traj - self.size_segment, size=num_init, replace=True
        ).reshape(-1, 1)
        time_index_2 = time_index + np.random.choice(
            len_traj - self.size_segment, size=num_init, replace=True
        ).reshape(-1, 1)

        sa_t_1 = np.take(sa_t_1, time_index_1, axis=0)
        r_t_1 = np.take(r_t_1, time_index_1, axis=0)
        sa_t_2 = np.take(sa_t_2, time_index_2, axis=0)
        r_t_2 = np.take(r_t_2, time_index_2, axis=0)

        # per-pair on-policy weights (segment-level)
        if policy_log_probs is not None and len(policy_log_probs) > 0:
            num_init = sa_t_1.shape[0]
            starts_1 = (time_index_1[:, 0] - np.arange(num_init) * len_traj)
            starts_2 = (time_index_2[:, 0] - np.arange(num_init) * len_traj)

            seg_logsum_1 = np.array([
                np.sum(policy_log_probs[batch_index_1[i]][
                       starts_1[i]:starts_1[i] + self.size_segment])
                for i in range(num_init)
            ], dtype=float)
            seg_logsum_2 = np.array([
                np.sum(policy_log_probs[batch_index_2[i]][
                       starts_2[i]:starts_2[i] + self.size_segment])
                for i in range(num_init)
            ], dtype=float)

            all_seg = np.concatenate([seg_logsum_1, seg_logsum_2])
            mu = all_seg.mean()
            sigma = all_seg.std()
            if sigma > 0:
                z1 = np.maximum(0.0, (seg_logsum_1 - mu) / sigma)
                z2 = np.maximum(0.0, (seg_logsum_2 - mu) / sigma)
            else:
                z1 = np.zeros_like(seg_logsum_1)
                z2 = np.zeros_like(seg_logsum_2)

            on_policy_weights = np.sqrt(z1 * z2)
            maxw = on_policy_weights.max() if on_policy_weights.size > 0 else 0.0
            if maxw > 0:
                on_policy_weights = on_policy_weights / maxw
            else:
                on_policy_weights = np.ones_like(on_policy_weights)
        else:
            on_policy_weights = np.ones(sa_t_1.shape[0], dtype=float)

        # ----------------- Stage 2: Uncertain query selection (ξU) -----------------
        uncertainties, consensus_mask = self.get_epistemic_uncertainty(sa_t_1, sa_t_2)
        # (you can swap this with other uncertainty variants if you want)

        if np.sum(consensus_mask) == 0:
            num_uncertain = min(self.mb_size * 2, num_init)
            top_uncertain_idx = (-uncertainties).argsort()[:num_uncertain]
        else:
            non_consensual_idx = np.where(consensus_mask)[0]
            non_cons_unc = uncertainties[consensus_mask]
            num_uncertain = min(self.mb_size * 2, len(non_consensual_idx))
            sorted_idx = (-non_cons_unc).argsort()[:num_uncertain]
            top_uncertain_idx = non_consensual_idx[sorted_idx]

        sa_t_1 = sa_t_1[top_uncertain_idx]
        sa_t_2 = sa_t_2[top_uncertain_idx]
        r_t_1 = r_t_1[top_uncertain_idx]
        r_t_2 = r_t_2[top_uncertain_idx]
        on_policy_weights = on_policy_weights[top_uncertain_idx]

        # ----------------- Stage 3: Diverse query selection (ξD) -----------------
        # mode = getattr(self.cfg, "duo_stage3", "baseline")  # "baseline" or "latent"
        # if mode == "latent" and getattr(self, "latent_encoder", None) is not None:
        #     print("[DUO] Stage3 using LATENT features")
        #     z1 = self.encode_segments_latent(sa_t_1, grad=False)
        #     z2 = self.encode_segments_latent(sa_t_2, grad=False)
        #     u = torch.cat([z1, z2, z1 - z2], dim=-1)
        #     features = u.detach().cpu().numpy()
        # else:
        #     print("[DUO] Stage3 using BASELINE reward_diffs")
        #     reward_diffs = self.get_reward_differences(sa_t_1, sa_t_2)
        #     features = reward_diffs

        mode = getattr(self.cfg, "duo_stage3", "baseline")  # "baseline", "latent", "latent_hybrid"
        reward_lambda = getattr(self.cfg, "duo_reward_lambda", 0.1)
        print("mode is:", mode)
        print("latent_encoder is None? ->", getattr(self, "latent_encoder", None) is None)
        print("latent_encoder object:", getattr(self, "latent_encoder", None))

        # if mode == "latent" and getattr(self, "latent_encoder", None) is not None:
        #     # --- pure latent DUO: [z1, z2, z1 - z2] ---
        #     print("latent mode: ", mode)
        #     print("[DUO] Stage3 using LATENT features")
        #     z1 = self.encode_segments_latent(sa_t_1, grad=False)   # (N, d)
        #     z2 = self.encode_segments_latent(sa_t_2, grad=False)   # (N, d)
        #     u  = torch.cat([z1, z2, z1 - z2], dim=-1)              # (N, 3d)
        #     features = u.detach().cpu().numpy()

        # elif mode in ("latent_hybrid", "latent_reward") and getattr(self, "latent_encoder", None) is not None:
        #     # --- hybrid: latent geometry + reward difference term ---
        #     print("hybrid mode: ", mode)
        #     print("[DUO] Stage3 using LATENT + REWARD_DIFF features")

        #     # 1) latent part: [z1, z2, z1 - z2]
        #     z1 = self.encode_segments_latent(sa_t_1, grad=False)   # (N, d)
        #     z2 = self.encode_segments_latent(sa_t_2, grad=False)   # (N, d)
        #     u  = torch.cat([z1, z2, z1 - z2], dim=-1)              # (N, 3d)
        #     latent_feats = u.detach().cpu().numpy()                # (N, 3d)

        #     # 2) reward difference in original space (using *true* segment rewards)
        #     # r_t_* shape: (N, size_segment, 1)
        #     sum_r1 = r_t_1.sum(axis=1)    # (N, 1)
        #     sum_r2 = r_t_2.sum(axis=1)    # (N, 1)
        #     delta_r = sum_r1 - sum_r2     # (N, 1), signed reward diff

        #     # normalize ΔR to avoid blowing up K-means
        #     mu = delta_r.mean()
        #     sigma = delta_r.std()
        #     if sigma > 1e-8:
        #         delta_r_norm = (delta_r - mu) / sigma
        #     else:
        #         delta_r_norm = np.zeros_like(delta_r)

        #     # 3) concat latent + scaled reward diff: [latent, λ * ΔR_norm]
        #     hybrid_feats = np.concatenate(
        #         [latent_feats, reward_lambda * delta_r_norm],
        #         axis=1
        #     )  # (N, 3d + 1)

        #     features = hybrid_feats

        # else:
        #     # --- baseline DUO Stage3: reward_diffs only ---
        #     print("[DUO] Stage3 using BASELINE reward_diffs")
        #     reward_diffs = self.get_reward_differences(sa_t_1, sa_t_2)
        #     features = reward_diffs



        if mode == "latent" and getattr(self, "latent_encoder", None) is not None:
            print("[DUO] Stage3 using LATENT features")

            # Encode both segments
            z1 = self.encode_segments_latent(sa_t_1, grad=False)   # (N, d)
            z2 = self.encode_segments_latent(sa_t_2, grad=False)   # (N, d)
            print("z1 shape: ", z1.shape)
            print("z2 shape: ", z2.shape)
            # Base DUO feature: [z1, z2, z1 - z2]
            u = torch.cat([z1, z2, z1 - z2], dim=-1)               # (N, 3d)
            print("u shape: ", u.shape)
            # NEW: cosine similarity between z1 and z2 as an extra scalar feature
            cos_sim = F.cosine_similarity(z1, z2, dim=-1)          # (N,)
            cos_sim = cos_sim.unsqueeze(1)                         # (N, 1)
            print("cos_sim shape: ", cos_sim.shape)
            # Final feature: [z1, z2, z1 - z2, cos(z1,z2)]
            u_with_cos = torch.cat([u, cos_sim], dim=-1)           # (N, 3d+1)

            features = u_with_cos.detach().cpu().numpy()
        else:
            print("[DUO] Stage3 using BASELINE reward_diffs")
            reward_diffs = self.get_reward_differences(sa_t_1, sa_t_2)
            features = reward_diffs


        num_diverse = min(self.mb_size, len(features))
        if len(features) <= num_diverse:
            selected_index = list(range(len(features)))
        else:
            selected_index, k_used = self.adaptive_kmeans_clustering(
                features, max_k=num_diverse
            )
            print(f"[DUO] adaptive_kmeans selected K={k_used}, "
                  f"from {len(features)} candidates")

        sa_t_1 = sa_t_1[selected_index]
        sa_t_2 = sa_t_2[selected_index]
        r_t_1 = r_t_1[selected_index]
        r_t_2 = r_t_2[selected_index]

        # ----------------- Get labels from teacher -----------------
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2
        )

        # ----------------- Online latent InfoNCE update -----------------
        if (
            getattr(self, "latent_encoder", None) is not None and
            getattr(self.cfg, "duo_stage3", "baseline") == "latent"
        ):
            labels_flat = labels.flatten()
            valid_mask = labels_flat != -1   # drop ties
            num_valid = int(valid_mask.sum())

            if num_valid >= 2:
                sa1_valid = sa_t_1[valid_mask]
                sa2_valid = sa_t_2[valid_mask]
                y_valid_np = labels_flat[valid_mask]

                # encode with gradients
                z1 = self.encode_segments_latent(sa1_valid, grad=True)
                z2 = self.encode_segments_latent(sa2_valid, grad=True)

                # build anchor / pos / neg from preference label
                y_valid = torch.from_numpy(y_valid_np).long().to(self.latent_device)
                mask_pos_seg2 = (y_valid == 1).unsqueeze(1)  # (B,1)

                z_pos = torch.where(mask_pos_seg2, z2, z1)
                z_neg = torch.where(mask_pos_seg2, z1, z2)
                z_anchor = z_pos  # anchor = preferred seg

                info_loss = self.train_latent_encoder(z_anchor, z_pos, z_neg)
                print(f"[DUO] Latent InfoNCE step: loss={info_loss:.4f}, "
                      f"valid_pairs={num_valid}")
            else:
                print("[DUO] Latent InfoNCE: not enough non-tie pairs this step")

        # ----------------- Store labeled queries -----------------
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)

        return len(labels)





    def train_reward(self):
        ensemble_losses = [[] for _ in range(self.de)]
        ensemble_acc = np.array([0 for _ in range(self.de)])
        
        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = []
        for _ in range(self.de):
            total_batch_index.append(np.random.permutation(max_len))
        
        num_epochs = int(np.ceil(max_len/self.train_batch_size))
        list_debug_loss1, list_debug_loss2 = [], []
        total = 0
        
        for epoch in range(num_epochs):
            self.opt.zero_grad()
            loss = 0.0
            
            last_index = (epoch+1)*self.train_batch_size
            if last_index > max_len:
                last_index = max_len
                
            for member in range(self.de):
                
                # get random batch
                idxs = total_batch_index[member][epoch*self.train_batch_size:last_index]
                sa_t_1 = self.buffer_seg1[idxs]
                sa_t_2 = self.buffer_seg2[idxs]
                labels = self.buffer_label[idxs]
                labels = torch.from_numpy(labels.flatten()).long().to(device)
                
                if member == 0:
                    total += labels.size(0)
                
                # get logits
                r_hat1 = self.r_hat_member(sa_t_1, member=member)
                r_hat2 = self.r_hat_member(sa_t_2, member=member)
                r_hat1 = r_hat1.sum(axis=1)
                r_hat2 = r_hat2.sum(axis=1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)

                # compute loss
                curr_loss = self.CEloss(r_hat, labels)
                loss += curr_loss
                ensemble_losses[member].append(curr_loss.item())
                
                # compute acc
                _, predicted = torch.max(r_hat.data, 1)
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct
                
            loss.backward()
            self.opt.step()
        
        ensemble_acc = ensemble_acc / total
        
        return ensemble_acc
    
    def train_soft_reward(self):
        ensemble_losses = [[] for _ in range(self.de)]
        ensemble_acc = np.array([0 for _ in range(self.de)])
        
        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = []
        for _ in range(self.de):
            total_batch_index.append(np.random.permutation(max_len))
        
        num_epochs = int(np.ceil(max_len/self.train_batch_size))
        list_debug_loss1, list_debug_loss2 = [], []
        total = 0
        
        for epoch in range(num_epochs):
            self.opt.zero_grad()
            loss = 0.0
            
            last_index = (epoch+1)*self.train_batch_size
            if last_index > max_len:
                last_index = max_len
                
            for member in range(self.de):
                
                # get random batch
                idxs = total_batch_index[member][epoch*self.train_batch_size:last_index]
                sa_t_1 = self.buffer_seg1[idxs]
                sa_t_2 = self.buffer_seg2[idxs]
                labels = self.buffer_label[idxs]
                labels = torch.from_numpy(labels.flatten()).long().to(device)
                
                if member == 0:
                    total += labels.size(0)
                
                # get logits
                r_hat1 = self.r_hat_member(sa_t_1, member=member)
                r_hat2 = self.r_hat_member(sa_t_2, member=member)
                r_hat1 = r_hat1.sum(axis=1)
                r_hat2 = r_hat2.sum(axis=1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)

                # compute loss
                uniform_index = labels == -1
                labels[uniform_index] = 0
                target_onehot = torch.zeros_like(r_hat).scatter(1, labels.unsqueeze(1), self.label_target)
                target_onehot += self.label_margin
                if sum(uniform_index) > 0:
                    target_onehot[uniform_index] = 0.5
                curr_loss = self.softXEnt_loss(r_hat, target_onehot)
                loss += curr_loss
                ensemble_losses[member].append(curr_loss.item())
                
                # compute acc
                _, predicted = torch.max(r_hat.data, 1)
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct
                
            loss.backward()
            self.opt.step()
        
        ensemble_acc = ensemble_acc / total
        
        return ensemble_acc