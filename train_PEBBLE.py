#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import os
import sys
import time
import pickle as pkl
import tqdm
from hydra.utils import get_original_cwd

from logger import Logger
from replay_buffer import ReplayBuffer
from reward_model import RewardModel
from collections import deque

import utils
import hydra

try:
    import wandb
except Exception:
    wandb = None





class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        self.root_dir = get_original_cwd()
        print(f"workspace: {self.work_dir}")
        print(f"project root: {self.root_dir}")

        print(f'workspace: {self.work_dir}')


        self.cfg = cfg

        print("DEBUG: cfg.seed =", getattr(cfg, "seed", None))
        print("DEBUG: ENV SEED =", os.environ.get("SEED"))

        # --- W&B init (optional, controlled by Hydra cfg.wandb.*) ---
        self.wandb_run = None
        if getattr(cfg, "wandb", None) and getattr(cfg.wandb, "enabled", True) and wandb is not None:
            from omegaconf import OmegaConf

            wb_mode   = getattr(cfg.wandb, "mode", "online")
            wb_proj   = getattr(cfg.wandb, "project", "Active_Querying_Techniques")
            wb_entity = getattr(cfg.wandb, "entity", "Uniform Sampling")
            wb_group  = getattr(cfg.wandb, "group", "Uniform Sampling") or cfg.env
            wb_name   = getattr(cfg.wandb, "name", None) or f"{cfg.env}-{cfg.seed}"

            wb_cfg = OmegaConf.to_container(cfg, resolve=True)

            # before (likely)
            # self.wandb_run = wandb.init(
            #     project="BPref_Active_Querying",
            #     group="Original Sampling",
            #     name=str(cfg.seed),            # <- int! causes ValidationError
            #     config=OmegaConf.to_container(cfg, resolve=True), job_type = "eval"
            # )

            # after
            # robust feed_type -> scheme mapping (Hydra may give strings)
            scheme_map = {
                0: "Uniform",
                1: "Disagreement",
                2: "Entropy",
                3: "KCenter",
                4: "KCenter_Disagree",
                5: "KCenter_Entropy",
                6: "DUO",
            }

            # normalize feed_type to int if possible, else None
            _raw_feed = getattr(cfg, "feed_type", None)
            try:
                feed_type_val = int(_raw_feed) if _raw_feed is not None else None
            except Exception:
                feed_type_val = None

            scheme = scheme_map.get(feed_type_val, "Baseline")

            # ensure seed string is safe for naming
            seed_val = getattr(cfg, "seed", "unknown")
            run_name = f"{scheme}_{seed_val}"

            self.wandb_run = wandb.init(
                project="BPref_Active_Querying",
                group="Online Latent - all + cosine similarity",
                name=run_name,
                config=OmegaConf.to_container(cfg, resolve=True),
                reinit=True,  # optional but handy for repeated runs in same process
            )

            try:
                wandb.define_metric("step")
                wandb.define_metric("*", step_metric="step")
            except Exception:
                pass




        self.logger = Logger(
            self.work_dir,
            save_tb=cfg.log_save_tb,
            log_frequency=cfg.log_frequency,
            agent="sac",
        )

        # Mirror logger logs to W&B with a global "step"
        if self.wandb_run is not None:
            _orig_log = self.logger.log
            def _wb_log(tag, value, step):
                try:
                    wandb.log({tag: value, "step": step})
                except Exception:
                    pass
                return _orig_log(tag, value, step)
            self.logger.log = _wb_log


        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.log_success = False
        
        # make env
        if 'metaworld' in cfg.env:
            self.env = utils.make_metaworld_env(cfg)
            self.log_success = True
        else:
            self.env = utils.make_env(cfg)
        
        cfg.agent.params.obs_dim = self.env.observation_space.shape[0]
        cfg.agent.params.action_dim = self.env.action_space.shape[0]
        cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
#         agent_cfg = cfg.agent
# # Expand params as kwargs so SACAgent(**params) is called, and avoid passing a `params=` kwarg.
#         from omegaconf import OmegaConf

        from omegaconf import OmegaConf

        agent_cfg = cfg.agent
        # Resolve interpolations and convert nested DictConfigs to plain dicts
        agent_params = OmegaConf.to_container(agent_cfg.params, resolve=True)
        self.agent = hydra.utils.instantiate({"_target_": agent_cfg._target_}, **agent_params)


        self.replay_buffer = ReplayBuffer(
            self.env.observation_space.shape,
            self.env.action_space.shape,
            int(cfg.replay_buffer_capacity),
            self.device)
        self.traj_records = []          # list of per-step dicts
        self.current_episode_id = 0        
        # for logging
        self.total_feedback = 0
        self.labeled_feedback = 0
        self.step = 0

        # instantiating the reward model
        self.reward_model = RewardModel(
            self.env.observation_space.shape[0],
            self.env.action_space.shape[0],
            ensemble_size=cfg.ensemble_size,
            size_segment=cfg.segment,
            activation=cfg.activation, 
            lr=cfg.reward_lr,
            mb_size=cfg.reward_batch, 
            large_batch=cfg.large_batch, 
            label_margin=cfg.label_margin, 
            teacher_beta=cfg.teacher_beta, 
            teacher_gamma=cfg.teacher_gamma, 
            teacher_eps_mistake=cfg.teacher_eps_mistake, 
            teacher_eps_skip=cfg.teacher_eps_skip, 
            teacher_eps_equal=cfg.teacher_eps_equal,
            cfg=self.cfg
        )

        # ---- Load latent encoder from project root ----
        autoenc_ckpt = os.path.join(self.root_dir, "latent_models", "autoencoder.pth")
        print(f"[LatentEncoder] Trying to load: {autoenc_ckpt}")
        print(f"[LatentEncoder] Path exists? {os.path.exists(autoenc_ckpt)}")

        try:
            self.reward_model.load_latent_encoder(autoenc_ckpt, self.device)
            print("[LatentEncoder] After load, latent_encoder is None? ->",
                  getattr(self.reward_model, 'latent_encoder', None) is None)
        except Exception as e:
            print(f"[LatentEncoder] Failed to load encoder from {autoenc_ckpt}: {e}")


        
    def evaluate(self):
        print(f"Starting evaluation at step {self.step}", flush=True)
        print(self.device)

        average_episode_reward = 0
        average_true_episode_reward = 0
        success_rate = 0
        
        for episode in range(self.cfg.num_eval_episodes):
            obs = self.env.reset()
            self.agent.reset()
            done = False
            episode_reward = 0
            true_episode_reward = 0
            if self.log_success:
                episode_success = 0

            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, done, extra = self.env.step(action)
                
                episode_reward += reward
                true_episode_reward += reward
                if self.log_success:
                    episode_success = max(episode_success, extra['success'])
                
            average_episode_reward += episode_reward
            average_true_episode_reward += true_episode_reward
            if self.log_success:
                success_rate += episode_success
            
        average_episode_reward /= self.cfg.num_eval_episodes
        average_true_episode_reward /= self.cfg.num_eval_episodes
        if self.log_success:
            success_rate /= self.cfg.num_eval_episodes
            success_rate *= 100.0
        
        self.logger.log('eval/episode_reward', average_episode_reward,
                        self.step)
        self.logger.log('eval/true_episode_reward', average_true_episode_reward,
                        self.step)
        if self.log_success:
            self.logger.log('eval/success_rate', success_rate,
                    self.step)
            self.logger.log('train/true_episode_success', success_rate,
                        self.step)
        self.logger.dump(self.step)
        print(f"[DEBUG] Finished evaluation at step {self.step}", flush=True)

    
    def learn_reward(self, first_flag=0):
        print("[DEBUG] cfg.feed_type =", getattr(self.cfg, "feed_type", None))
        feed_type = getattr(self.cfg, "feed_type", None)
        print("[DEBUG] feed_type at runtime:", self.cfg.feed_type)

        no_sampling = feed_type is None 

        labeled_queries = 0
        if no_sampling:
            print("Baseline!")
        if not no_sampling:
            if first_flag == 1:
                print("[DEBUG] Using uniform_sampling for first_flag==1")
                labeled_queries = self.reward_model.uniform_sampling()
            else:
                if self.cfg.feed_type == 0:
                    print("[DEBUG] Using uniform_sampling for feed_type==0")
                    labeled_queries = self.reward_model.uniform_sampling()
                elif self.cfg.feed_type == 1:
                    print("[DEBUG] Using disagreement_sampling for feed_type==1")
                    labeled_queries = self.reward_model.disagreement_sampling()
                elif self.cfg.feed_type == 2:
                    print("[DEBUG] Using entropy_sampling for feed_type==2")
                    labeled_queries = self.reward_model.entropy_sampling()
                elif self.cfg.feed_type == 3:
                    print("[DEBUG] Using kcenter_sampling for feed_type==3")
                    labeled_queries = self.reward_model.kcenter_sampling()
                elif self.cfg.feed_type == 4:
                    print("[DEBUG] Using kcenter_disagree_sampling for feed_type==4")
                    labeled_queries = self.reward_model.kcenter_disagree_sampling()
                elif self.cfg.feed_type == 5:
                    print("[DEBUG] Using kcenter_entropy_sampling for feed_type==5")
                    labeled_queries = self.reward_model.kcenter_entropy_sampling()
                elif self.cfg.feed_type == 6:
                    print("[DEBUG] Using duo_sampling for feed_type==6")
                    # DUO sampling - without policy log probs (can be extended later)
                    labeled_queries = self.reward_model.duo_sampling(policy_log_probs=None)
                else:
                    raise NotImplementedError
        else:
            labeled_queries = 0

        self.total_feedback += labeled_queries
        self.labeled_feedback += labeled_queries

        train_acc = 0
        if self.labeled_feedback > 0:
            for epoch in range(self.cfg.reward_update):
                if self.cfg.label_margin > 0 or self.cfg.teacher_eps_equal > 0:
                    train_acc = self.reward_model.train_soft_reward()
                else:
                    train_acc = self.reward_model.train_reward()
                total_acc = np.mean(train_acc)
                if total_acc > 0.97:
                    break

            print("Reward function is updated!! ACC: " + str(total_acc))
        else:
            print("No labeled feedback available; reward not updated.")

    def export_trajectory_dataset(self, path):
        """
        Convert self.traj_records to arrays and save them as a .npz file.

        Result shapes:
            obs:        (N, *obs_dim)
            action:     (N, *action_dim)
            reward:     (N,)
            done:       (N,)
            episode_id: (N,)
            timestep:   (N,)
        """
        if len(self.traj_records) == 0:
            print("[TrajectoryDataset] No records to save.")
            return

        obs = np.stack([rec["obs"] for rec in self.traj_records], axis=0)
        action = np.stack([rec["action"] for rec in self.traj_records], axis=0)
        reward = np.array([rec["reward"] for rec in self.traj_records], dtype=np.float32)
        done = np.array([rec["done"] for rec in self.traj_records], dtype=np.bool_)
        episode_id = np.array([rec["episode_id"] for rec in self.traj_records], dtype=np.int32)
        timestep = np.array([rec["timestep"] for rec in self.traj_records], dtype=np.int32)

        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(
            path,
            obs=obs,
            action=action,
            reward=reward,
            done=done,
            episode_id=episode_id,
            timestep=timestep,
        )
        print(f"[TrajectoryDataset] Saved {len(self.traj_records)} transitions to {path}")


    def export_preference_dataset(self, path):
        """
        Save all stored preference queries to disk for offline training.

        path: str or Path, e.g. 'datasets/preferences_prefRL.npz'

        Stored shapes:
            seg1:   (N, size_segment, ds+da)
            seg2:   (N, size_segment, ds+da)
            label:  (N,)     in {0, 1, -1}
        """
        # How many queries do we actually have?
        max_len = self.reward_model.capacity if self.reward_model.buffer_full else self.reward_model.buffer_index

        seg1 = self.reward_model.buffer_seg1[:max_len].astype(np.float32)
        seg2 = self.reward_model.buffer_seg2[:max_len].astype(np.float32)
        labels = self.reward_model.buffer_label[:max_len].astype(np.int8).reshape(-1)

        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(
            path,
            seg1=seg1,
            seg2=seg2,
            label=labels,
            size_segment=self.reward_model.size_segment,
            ds=self.reward_model.ds,
            da=self.reward_model.da,
        )
        print(f"[RewardModel] Saved {max_len} preference queries to {path}")





    def run(self):
        episode, episode_reward, done = 0, 0, True
        if self.log_success:
            episode_success = 0
        true_episode_reward = 0
        
        # store train returns of recent 10 episodes
        avg_train_true_return = deque([], maxlen=10) 
        start_time = time.time()

        interact_count = 0
        while self.step < self.cfg.num_train_steps:
            if done:
                if self.step > 0:
                    self.logger.log('train/duration', time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(
                        self.step, save=(self.step > self.cfg.num_seed_steps))

                # evaluate agent periodically
                if self.step > 0 and self.step % self.cfg.eval_frequency == 0:
                    self.logger.log('eval/episode', episode, self.step)
                    self.evaluate()
                
                self.logger.log('train/episode_reward', episode_reward, self.step)
                self.logger.log('train/true_episode_reward', true_episode_reward, self.step)
                self.logger.log('train/total_feedback', self.total_feedback, self.step)
                self.logger.log('train/labeled_feedback', self.labeled_feedback, self.step)
                
                if self.log_success:
                    self.logger.log('train/episode_success', episode_success,
                        self.step)
                    self.logger.log('train/true_episode_success', episode_success,
                        self.step)
                
                obs = self.env.reset()
                self.agent.reset()
                done = False
                episode_reward = 0
                avg_train_true_return.append(true_episode_reward)
                true_episode_reward = 0
                if self.log_success:
                    episode_success = 0
                episode_step = 0
                episode += 1
                
                self.current_episode_id += 1

                self.logger.log('train/episode', episode, self.step)
                        
            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # run training update                
            if self.step == (self.cfg.num_seed_steps + self.cfg.num_unsup_steps):
                # update schedule
                if self.cfg.reward_schedule == 1:
                    frac = (self.cfg.num_train_steps-self.step) / self.cfg.num_train_steps
                    if frac == 0:
                        frac = 0.01
                elif self.cfg.reward_schedule == 2:
                    frac = self.cfg.num_train_steps / (self.cfg.num_train_steps-self.step +1)
                else:
                    frac = 1
                self.reward_model.change_batch(frac)
                
                # update margin --> not necessary / will be updated soon
                new_margin = np.mean(avg_train_true_return) * (self.cfg.segment / self.env._max_episode_steps)
                self.reward_model.set_teacher_thres_skip(new_margin)
                self.reward_model.set_teacher_thres_equal(new_margin)
                
                # first learn reward
                self.learn_reward(first_flag=0)
                
                # relabel buffer
                self.replay_buffer.relabel_with_predictor(self.reward_model)
                
                # reset Q due to unsuperivsed exploration
                self.agent.reset_critic()
                
                # update agent
                self.agent.update_after_reset(
                    self.replay_buffer, self.logger, self.step, 
                    gradient_update=self.cfg.reset_update, 
                    policy_update=True)
                
                # reset interact_count
                interact_count = 0
            elif self.step > self.cfg.num_seed_steps + self.cfg.num_unsup_steps:
                # update reward function
                if self.total_feedback < self.cfg.max_feedback:
                    if interact_count == self.cfg.num_interact:
                        # update schedule
                        if self.cfg.reward_schedule == 1:
                            frac = (self.cfg.num_train_steps-self.step) / self.cfg.num_train_steps
                            if frac == 0:
                                frac = 0.01
                        elif self.cfg.reward_schedule == 2:
                            frac = self.cfg.num_train_steps / (self.cfg.num_train_steps-self.step +1)
                        else:
                            frac = 1
                        self.reward_model.change_batch(frac)
                        
                        # update margin --> not necessary / will be updated soon
                        new_margin = np.mean(avg_train_true_return) * (self.cfg.segment / self.env._max_episode_steps)
                        self.reward_model.set_teacher_thres_skip(new_margin * self.cfg.teacher_eps_skip)
                        self.reward_model.set_teacher_thres_equal(new_margin * self.cfg.teacher_eps_equal)
                        
                        # corner case: new total feed > max feed
                        if self.reward_model.mb_size + self.total_feedback > self.cfg.max_feedback:
                            self.reward_model.set_batch(self.cfg.max_feedback - self.total_feedback)
                            
                        self.learn_reward()
                        self.replay_buffer.relabel_with_predictor(self.reward_model)
                        interact_count = 0
                        
                self.agent.update(self.replay_buffer, self.logger, self.step, 1)
                
            # unsupervised exploration
            elif self.step > self.cfg.num_seed_steps:
                self.agent.update_state_ent(self.replay_buffer, self.logger, self.step, 
                                            gradient_update=1, K=self.cfg.topK)
                
            next_obs, reward, done, extra = self.env.step(action)
            reward_hat = self.reward_model.r_hat(np.concatenate([obs, action], axis=-1))
            self.traj_records.append(
                {
                    "episode_id": self.current_episode_id,
                    "timestep": episode_step,
                    "obs": np.array(obs, copy=True),
                    "action": np.array(action, copy=True),
                    "reward": float(reward),
                    "done": bool(done),
                }
            )
            
            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            episode_reward += reward_hat
            true_episode_reward += reward
            
            if self.log_success:
                episode_success = max(episode_success, extra['success'])
                
            # adding data to the reward training data
            self.reward_model.add_data(obs, action, reward, done)
            self.replay_buffer.add(
                obs, action, reward_hat, 
                next_obs, done, done_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1
            interact_count += 1
            # ---- DEBUG: progress print every 1000 steps ----
            if self.step % 1000 == 0:
                print(
                    f"[DEBUG] step={self.step} episode={episode} "
                    f"ep_reward={episode_reward:.2f} true_ep_reward={true_episode_reward:.2f}",
                    flush=True,
                )
            # -------------------------------------------------

        # ---- NEW: dump offline datasets (trajectory + preferences) ----
        datasets_dir = os.path.join(self.work_dir, "datasets")
        os.makedirs(datasets_dir, exist_ok=True)

        traj_path = os.path.join(datasets_dir, f"traj_seed{self.cfg.seed}.npz")
        self.export_trajectory_dataset(traj_path)

        # pref_path = os.path.join(datasets_dir, f"prefs_seed{self.cfg.seed}.npz")
        # try:
        #     self.reward_model.export_preference_dataset(pref_path)
        # except AttributeError:
        #     print("[Warning] RewardModel has no export_preference_dataset(). "
        #           "Did you add it to reward_model.py?")

        pref_path = os.path.join(datasets_dir, f"prefs_seed{self.cfg.seed}.npz")
        self.export_preference_dataset(pref_path)

        # ----------------------------------------------------------------
            
        self.agent.save(self.work_dir, self.step)
        self.reward_model.save(self.work_dir, self.step)
        # --- W&B artifacts + graceful finish ---
        if self.wandb_run is not None:
            try:
                art = wandb.Artifact(f"{self.cfg.env}-pebble-models", type="model")

                # common saved files (add if they exist)
                candidates = [
                    os.path.join(self.work_dir, f"actor_{self.step}.pt"),
                    os.path.join(self.work_dir, f"critic_{self.step}.pt"),
                    os.path.join(self.work_dir, f"critic_target_{self.step}.pt"),
                    os.path.join(self.work_dir, f"reward_{self.step}.pt"),
                ]
                for p in candidates:
                    if os.path.exists(p):
                        art.add_file(p)

                wandb.log_artifact(art)
            except Exception:
                pass
            finally:
                try:
                    wandb.finish()
                except Exception:
                    pass

        
@hydra.main(version_base=None, config_path='config', config_name='train_PEBBLE')


def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()

if __name__ == '__main__':
    main()