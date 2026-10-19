"""Shared four-leg screw policy with the agreed fixed-arm skill adapters.

Rewards and observations remain inherited. Termination adds the approved reach
box and two-second whole-hand/object contact-loss timer.
"""
import json,time
from pathlib import Path
import numpy as np
import torch
from gymnasium import spaces
from isaacsimenvs.tasks.precise_assembly.precise_assembly_env import PreciseAssemblyEnv
from sgs import SuccessGuidedSampler
from reachability import boxes_disjoint, quaternion_matrix
from contact_loss import HandObjectContactLoss

class ScrewEnv(PreciseAssemblyEnv):
 _enable_hand_object_contacts=True
 def __init__(self,cfg,work,**kw):
  self.work=Path(work);self.skill_ready=False
  super().__init__(cfg,**kw)
  self.hids=self.robot.find_joints(list(cfg.robot.joint_order)[7:],preserve_order=True)[0]
  self.aids=self.robot.find_joints(list(cfg.robot.joint_order)[:7],preserve_order=True)[0]
  self.hlo=self.robot.data.joint_pos_limits[0,self.hids,0];self.hhi=self.robot.data.joint_pos_limits[0,self.hids,1]
  with np.load(self.work/'screw_resets.npz') as z:
   self.pool={k:torch.as_tensor(z[k],device=self.device,dtype=torch.long if k in ('progress','leg_id') else torch.float32) for k in ['arm_q','hand_q','hand_target','object_xyzw','progress','leg_id']}
  self.groups=[torch.where(self.pool['progress']==i)[0] for i in range(8)]
  spec=json.loads((self.work/'screw_goals.json').read_text());self.spec=spec
  self.leg_names=spec['objects']
  poses=torch.tensor([[e['representative_object_xyzw'] for e in inst['events']] for inst in spec['instances']],device=self.device)
  self.goals=torch.cat([poses[...,:3],poses[...,6:7],poses[...,3:6]],-1)
  self.fixture=torch.tensor(spec['scene']['fixture_xyzw'],device=self.device)
  self.start_progress=torch.zeros(self.num_envs,dtype=torch.long,device=self.device)
  self.active_leg=torch.zeros_like(self.start_progress)
  self.reset_row=torch.zeros_like(self.start_progress)
  self.arm_hold=torch.zeros(self.num_envs,7,device=self.device)
  self.completed=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
  self.started=time.time()
  self.sampler=SuccessGuidedSampler(len(self.pool['progress']), self.device)
  reach=json.loads((Path(__file__).parent/'reach_geometry.json').read_text())
  hb=torch.tensor(reach['hand_bounds_m'],device=self.device)
  ob=torch.tensor(reach['object_bounds_m'],device=self.device)
  self.reach_hand_center=hb.mean(0);self.reach_hand_half=(hb[1]-hb[0])/2
  self.reach_object_center=ob.mean(0);self.reach_object_half=(ob[1]-ob[0])/2
  mount=torch.tensor(reach['palm_mount_from_fr3_link7'],device=self.device)
  self.reach_mount_r=mount[:3,:3];self.reach_mount_p=mount[:3,3]
  self.reach_link=self.robot.find_bodies('fr3_link7')[0][0]
  self.contact_loss=HandObjectContactLoss(self,seconds=2.0)
  self.skill_ready=True
  cfg.action_space=12
  self.single_action_space=spaces.Box(-1.,1.,(12,),np.float32)
  self.action_space=spaces.Box(-1.,1.,(self.num_envs,12),np.float32)
  self.actions=torch.zeros(self.num_envs,12,device=self.device)
  self._reset_idx(torch.arange(self.num_envs,device=self.device))
  (self.work/'rl_environment.json').write_text(json.dumps(dict(reset_rows=len(self.pool['progress']),resets_per_leg={name:int((self.pool['leg_id']==i).sum()) for i,name in enumerate(self.leg_names)},resets_per_progress=[len(x) for x in self.groups],actor_dim=cfg.observation_space,critic_dim=cfg.state_space,actions=12,control_hz=1/self.step_dt,physics_hz=1/self.physics_dt,goal_source='our Stage 02/03 events for each sampled leg',reward='inherited PreciseAssemblyEnv._get_rewards',termination='base plus immediate reach-box separation and 2s whole-hand/object contact loss; fixture-force termination disabled',contact_loss_termination=self.contact_loss.metadata,observations='unchanged inherited world-axis/env-local observation construction',reset_sampler='global SGS over all four legs and all reset rows; no leg or stage hierarchy',started_unix=self.started),indent=2))

 @property
 def last_targets(self):return self._cur_targets[:,self.hids]

 def sample_reset_rows(self,count):
  rows=self.sampler.sample(count)
  return self.pool['progress'][rows],rows

 def _reset_idx(self,ids):
  if not self.skill_ready:return super()._reset_idx(ids)
  if ids is None:ids=torch.arange(self.num_envs,device=self.device)
  progress,rows=self.sample_reset_rows(len(ids))
  self.start_progress[ids]=progress
  self.active_leg[ids]=self.pool['leg_id'][rows]
  super()._reset_idx(ids)
  self.contact_loss.reset(ids)
  self.reset_row[ids]=rows;self.env_max_goals[ids]=8-progress;self.completed[ids]=False
  self.arm_hold[ids]=self.pool['arm_q'][rows]
  q=self.robot.data.default_joint_pos[ids].clone();q[:,self.aids]=self.arm_hold[ids];q[:,self.hids]=self.pool['hand_q'][rows]
  self.robot.write_joint_state_to_sim(q,torch.zeros_like(q),env_ids=ids)
  targets=q.clone();targets[:,self.hids]=self.pool['hand_target'][rows].clamp(self.hlo,self.hhi)
  self._cur_targets[ids]=targets;self._prev_targets[ids]=targets
  self.robot.set_joint_position_target(targets,env_ids=ids)
  op=self.pool['object_xyzw'][rows];root=torch.cat([op[:,:3]+self.scene.env_origins[ids],op[:,6:7],op[:,3:6]],-1)
  self.object.write_root_pose_to_sim(root,env_ids=ids);self.object.write_root_velocity_to_sim(torch.zeros(len(ids),6,device=self.device),env_ids=ids)
  fp=self.fixture.expand(len(ids),7);hp=torch.cat([fp[:,:3]+self.scene.env_origins[ids],fp[:,6:7],fp[:,3:6]],-1)
  self.hole_pos[ids]=fp[:,:3];self.hole_quat_wxyz[ids]=hp[:,3:]
  self.hole.write_root_pose_to_sim(hp,env_ids=ids);self.hole.write_root_velocity_to_sim(torch.zeros(len(ids),6,device=self.device),env_ids=ids)
  tp=torch.zeros(len(ids),7,device=self.device);tp[:,:3]=self.scene.env_origins[ids];tp[:,2]+=.38;tp[:,3]=1
  self.table.write_root_pose_to_sim(tp,env_ids=ids);self._table_z_per_env[ids]=.38
  self._lifted_object[ids]=True
  self._clear_goal_trackers(ids);self._write_goal_pose(ids,True)

 def _write_goal_pose(self,ids,is_first_goal=False):
  if not self.skill_ready:return super()._write_goal_pose(ids,is_first_goal)
  index=(self.start_progress[ids]+1+self._successes[ids]).clamp(max=8)
  g=self.goals[self.active_leg[ids],index].clone();g[:,:3]+=self.scene.env_origins[ids]
  self.goal_viz.write_root_pose_to_sim(g,env_ids=ids)
  self.goal_viz.write_root_velocity_to_sim(torch.zeros(len(ids),6,device=self.device),env_ids=ids)

 def _pre_physics_step(self,actions):
  self.contact_loss.begin()
  full=torch.zeros(self.num_envs,19,device=self.device);full[:,7:]=actions
  super()._pre_physics_step(full)
  self._cur_targets[:,self.aids]=self.arm_hold
  self._prev_targets[:,self.aids]=self.arm_hold

 def reach_disjoint(self):
  arm_r=quaternion_matrix(self.robot.data.body_quat_w[:,self.reach_link])
  palm_r=arm_r @ self.reach_mount_r
  palm_p=(self.robot.data.body_pos_w[:,self.reach_link]-self.scene.env_origins
          + arm_r @ self.reach_mount_p)
  obj_p=self.object.data.root_pos_w-self.scene.env_origins
  obj_r=quaternion_matrix(self.object.data.root_quat_w)
  return boxes_disjoint(palm_p,palm_r,obj_p,obj_r,
                       self.reach_hand_center,self.reach_hand_half,
                       self.reach_object_center,self.reach_object_half)

 def _get_dones(self):
  terminated,truncated=super()._get_dones()
  if not self.skill_ready:return terminated,truncated
  unreachable=self.reach_disjoint()
  self._termination_reasons['hand_unreachable']=unreachable
  contact_lost=self.contact_loss.update()
  self._termination_reasons['hand_object_contact_lost']=contact_lost
  terminated=terminated|unreachable|contact_lost
  self.completed=self._successes>=self.env_max_goals
  ended=terminated|truncated
  self.sampler.update(self.reset_row[ended],self.completed[ended])
  return terminated,truncated
