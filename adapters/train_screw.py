import argparse, os, sys, json, time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--num-envs',type=int,default=24576);p.add_argument('--epochs',type=int,default=1000000);p.add_argument('--smoke',action='store_true')
p.add_argument('--task-assets-root',type=Path,default=Path(__file__).resolve().parents[1]/'experiments/furniturebench_square_table/task')
p.add_argument('--run-name',default='furniturebench-square_table-contact_termination_2s')
p.add_argument('--wandb-project',default='simtoolreal')
p.add_argument('--wandb-entity',default=os.environ.get('WANDB_ENTITY','draftrec'))
p.add_argument('--viewer-raw-base',default='https://raw.githubusercontent.com/DAVIAN-Robotics/simtoolreal/furniturebench_square_table-omnireset_dexonomy-eigendexplore_dexonomy-sgs/')
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(p);args=p.parse_args();args.work=args.work.expanduser().resolve();args.task_assets_root=args.task_assets_root.expanduser().resolve();args.headless=True;app=AppLauncher(args).app
import torch,yaml
from isaaclab.envs.utils.spaces import replace_env_cfg_spaces_with_strings,replace_strings_with_env_cfg_spaces
from isaacsimenvs.tasks.precise_assembly.precise_assembly_env_cfg import PreciseAssemblyEnvCfg
from isaac_sysid import install,audit
install()
from reference_screw_env import ScrewEnv
from dataclasses import replace
from evaluation.problems import PROBLEM_REGISTRY
from evaluation.objects import NAME_TO_OBJECT
tree=Path(__file__).resolve().parents[1];os.chdir(tree)
source=args.task_assets_root
spec=json.loads((args.work/'screw_goals.json').read_text());scene=spec['scene'];obj=scene['objects'][0]
assert spec['objects']==['leg1','leg2','leg3','leg4']
assert len({o['urdf'] for o in scene['objects']})==1
problem=PROBLEM_REGISTRY['screwing']
NAME_TO_OBJECT[problem.insertion_object_name]=replace(NAME_TO_OBJECT[problem.insertion_object_name],urdf_path=source/'inputs'/obj['urdf'])
# Match the geometry used to validate the supplied event-state reset pool.
# One active leg is dynamic per environment, at its sampled socket.
# All four sockets remain open, as in the existing leg4 training run.
PROBLEM_REGISTRY['screwing']=replace(problem,receptive_urdf=str(source/'inputs'/scene['fixture_urdf']))
reference=yaml.safe_load((args.work/'september_config.yaml').read_text())['env']
cfg=replace_env_cfg_spaces_with_strings(PreciseAssemblyEnvCfg());cfg.from_dict(reference);cfg=replace_strings_with_env_cfg_spaces(cfg)
cfg.robot='fr3-xhand-a4hs';cfg.scene.num_envs=args.num_envs;cfg.precise_assembly.problems=('screwing',)
cfg.reset.table_reset_z_range=0.;cfg.precise_assembly.enable_retract=False
cfg.precise_assembly.hand_contact_penalty_enabled=False
cfg.precise_assembly.hand_force_termination_n=0.
# Explicit user decision: no observation/action delays, object-pose noise or
# random object force/torque disturbances for this experiment.
dr=cfg.domain_randomization
dr.use_obs_delay=False;dr.obs_delay_max=0
dr.use_action_delay=False;dr.action_delay_max=0
dr.use_object_state_delay_noise=False;dr.object_state_delay_max=0
dr.object_state_xyz_noise_std=0.;dr.object_state_rotation_noise_degrees=0.
dr.joint_velocity_obs_noise_std=0.;dr.object_scale_noise_multiplier_range=(1.,1.)
dr.force_scale=0.;dr.torque_scale=0.
cfg.precise_assembly.goal_xy_obs_noise=0.;cfg.precise_assembly.goal_yaw_obs_noise_deg=0.
# October's actor/privileged-critic fields; retain all other reference settings.
october=yaml.safe_load((args.work/'training_template.yaml').read_text())['env']
cfg.obs.obs_list=tuple(october['obs']['obs_list']);cfg.obs.state_list=tuple(october['obs']['state_list']);cfg.obs.keypoint_corner_ids=None
for name,value in reference['sim']['physx'].items():
 assert getattr(cfg.sim.physx,name)==value,(name,getattr(cfg.sim.physx,name),value)
(args.work/'resolved_environment_config.json').write_text(json.dumps(cfg.to_dict(),default=str,indent=2))
(args.work/'user_requested_exclusions.json').write_text(json.dumps(dict(observation_delay=False,action_delay=False,object_pose_noise=False,goal_pose_noise=False,joint_velocity_observation_noise=False,object_scale_observation_noise=False,random_object_wrenches=False),indent=2))
tick=time.time();env=ScrewEnv(cfg,work=args.work);audit(env,args.work/'rl_runtime_model.json');env.reset()
(args.work/'resolved_environment_config.json').write_text(json.dumps(cfg.to_dict(),default=str,indent=2))
if args.smoke:
 from verify_environment import verify
 verify(env,args.work)
 print('SMOKE_COMPLETE',flush=True);sys.stdout.flush();os._exit(0)
from rl_games.torch_runner import Runner
from isaacsimenvs.utils.rlgames_utils import register_rlgames_env,EnvStatsAlgoObserver,MultiObserver
from eigendexplore.rl_games import configure
from eigendexplore import EigenDExplore
import numpy as np
agent=yaml.safe_load((args.work/'training_template.yaml').read_text())['agent']
params=agent['params'];config=params['config'];config.update(name='0_'+args.run_name,full_experiment_name='0_'+args.run_name,num_actors=args.num_envs,expl_coef_block_size=args.num_envs//6,minibatch_size=args.num_envs*4,max_epochs=args.epochs,save_frequency=1000,save_best_after=100,train_dir=str(args.work/'training'),learning_rate=1e-4)
config['central_value_config'].update(minibatch_size=args.num_envs*4,learning_rate=1e-4)
params['load_checkpoint']=False;params['load_path']='';params['seed']=20261015
z=np.load(args.work/'eigen_dexonomy_xhand.npz');names=list(map(str,z['joint_names']));order=list(env.cfg.robot.joint_order)[7:];cols=[names.index(j) for j in order]
noise=EigenDExplore.from_pca(torch.tensor(z['components'][:,cols],dtype=torch.float64),torch.tensor(z['component_std'],dtype=torch.float64),rank=int(z['k']),eigen_rms=2**-.5,spectrum_power=1.,coordinate_scale=((env.hhi-env.hlo)/2).double().cpu())
configure(params,{'basis':noise.basis.tolist(),'eigen_sigma':noise.log_scale.detach().exp().tolist(),'iid_sigma':[2**-.5]*12})
(args.work/'training_config.json').write_text(json.dumps(agent,indent=2))
from isaacsimenvs.tasks.simtoolreal import pose_viewer
# Asset symlinks resolve to the supplied tree. Preserve the native URL mapping.
pose_viewer.REPO_ROOT=tree
env=pose_viewer.SimToolRealPoseViewerWrapper(env,output_dir=args.work/'interactive_viewer',
 capture_len=600,capture_interval=1000,env_id=0,wandb_key='interactive_viewer',
 github_raw_base=args.viewer_raw_base,url_check='error')
register_rlgames_env(env,rl_device='cuda:0',clip_obs=10.,clip_actions=1.)
from sampler_checkpoint import register_sampler_checkpoint
register_sampler_checkpoint(env.unwrapped)
from isaacsimenvs.utils.wandb_utils import WandbAlgoObserver
from omegaconf import OmegaConf
wandb_cfg=OmegaConf.create(dict(
 wandb_activate=True,wandb_project=args.wandb_project,wandb_entity=args.wandb_entity,
 wandb_name=args.run_name,wandb_exact_display_name=True,
 wandb_group=None,wandb_tags=[],wandb_notes='',wandb_logcode_dir=str(tree),
 wandb_prepared_diff=str(args.work/'source.patch'),
 env=json.loads((args.work/'resolved_environment_config.json').read_text()),agent=agent,
 reset_sampling=dict(method='SGS',pool_size=len(env.unwrapped.pool['progress']),history=100,target=.5,temperature=2.,
                     epsilon=1e-4,concentration=1.,empty_history_success_rate=0.,
                     objects=spec['objects'],scope='global across all reset rows, without leg or stage hierarchy',
                     success='completed remaining goals after reset'),
 contact_loss_termination=env.unwrapped.contact_loss.metadata,
 reach_termination=json.loads((tree/'adapters/reach_geometry.json').read_text())))
runner=Runner(MultiObserver([EnvStatsAlgoObserver(),WandbAlgoObserver(wandb_cfg)]))
runner.load(agent);runner.reset();print('TRAINING_FROM_SCRATCH',flush=True)
runner.run({'train':True,'play':False,'checkpoint':None})
(args.work/'training_complete.json').write_text(json.dumps(dict(time_unix=time.time(),epochs=args.epochs)))
sys.stdout.flush();os._exit(0)
