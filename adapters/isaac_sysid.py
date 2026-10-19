"""Install the linked training run's exact robot profile in this process only."""
import importlib.util,json,sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent

def install():
    path=ROOT/'audit/recent_training_robots.py'
    spec=importlib.util.spec_from_file_location('a4hs_source_robots',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    import isaacsimenvs.tasks.simtoolreal.robots as robots
    import isaacsimenvs.tasks.simtoolreal.utils.scene_utils as base
    import isaacsimenvs.tasks.precise_assembly.scene_utils as assembly
    profile=module.ROBOT_PROFILES['fr3-xhand-a4hs']
    robots.ROBOT_PROFILES['fr3-xhand-a4hs']=profile
    original=base.build_robot_articulation_usd_cfg
    def build(usd_path,robot,**kw):
        cfg=original(usd_path,robot,**kw)
        cfg.actuators['hand'].friction=robot.hand_friction
        cfg.actuators['hand'].dynamic_friction=robot.hand_dynamic_friction
        cfg.actuators['hand'].effort_limit_sim=robot.hand_effort_limit
        return cfg
    base.build_robot_articulation_usd_cfg=build
    assembly.build_robot_articulation_usd_cfg=build
    return profile

def audit(env,path):
    names=env.robot.joint_names
    profile=env.cfg.robot
    report=dict(robot='fr3-xhand-a4hs',joint_names=names,physics_dt=env.physics_dt,decimation=env.cfg.decimation)
    fields={'stiffness':'joint_stiffness','damping':'joint_damping','armature':'joint_armature','static_friction':'joint_friction_coeff','dynamic_friction':'joint_dynamic_friction_coeff','effort_limit':'joint_effort_limits'}
    for label,attr in fields.items():
        if not hasattr(env.robot.data,attr):
            raise RuntimeError(f'Missing runtime audit field {attr}')
        values=getattr(env.robot.data,attr).detach().cpu().numpy()
        np.testing.assert_allclose(values,np.broadcast_to(values[0],values.shape),atol=1e-7)
        report[label]=values[0].tolist()
    params=json.loads((ROOT/'hand_sysid.json').read_text())
    for name,p in params.items():
        i=names.index(name)
        for label in fields:
            # The JSON parameter order is Kp,Kd,static,dynamic,armature,effort.
            expected=p[{'stiffness':0,'damping':1,'static_friction':2,'dynamic_friction':3,'armature':4,'effort_limit':5}[label]]
            np.testing.assert_allclose(report[label][i],expected,rtol=1e-5,atol=1e-7,err_msg=name+' '+label)
    for label,values in [('stiffness',profile.arm_stiffness),('damping',profile.arm_damping),('armature',profile.arm_armature),('static_friction',profile.arm_friction)]:
        for name,expected in values.items():np.testing.assert_allclose(report[label][names.index(name)],expected,rtol=1e-5,atol=1e-7)
    report['assembly_contact_settings']={k:getattr(env.cfg.sim.physx,k) for k in ['friction_offset_threshold','friction_correlation_distance','solver_type','enable_stabilization','bounce_threshold_velocity']}
    report['scope']='Exact a4hs robot actuator profile; retains precise-assembly contact solver settings. Recent general-manipulation training uses different contact-patch grouping distances.'
    report['contact_materials']={k:getattr(env.cfg.assets,k) for k in ['robot_friction','finger_tip_friction','object_friction','table_friction']}
    assert report['contact_materials']==dict(robot_friction=.5,finger_tip_friction=1.5,object_friction=.5,table_friction=.5)
    report['actuator_configs']={k:{f:getattr(v.cfg,f,None) for f in ['stiffness','damping','armature','friction','dynamic_friction','effort_limit_sim']} for k,v in env.robot.actuators.items()}
    Path(path).write_text(json.dumps(report,indent=2))
    print('ISAAC_A4HS_PARAMETERS_VERIFIED',flush=True)
