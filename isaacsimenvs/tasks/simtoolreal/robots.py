from __future__ import annotations

from dataclasses import MISSING

from isaaclab.utils import configclass


@configclass
class RobotCfg:
    urdf: str = MISSING
    arm_joint_regex: str = MISSING
    hand_joint_regex: str = MISSING
    joint_order: tuple[str, ...] = MISSING
    palm_body: str = MISSING
    fingertip_body_regex: str = MISSING
    fingertip_bodies: tuple[str, ...] = MISSING
    arm_stiffness: dict[str, float] = MISSING
    arm_damping: dict[str, float] = MISSING
    hand_stiffness: dict[str, float] = MISSING
    hand_damping: dict[str, float] = MISSING
    hand_armature: dict[str, float] = MISSING
    arm_default_pos: dict[str, float] = MISSING
    palm_center_offset: tuple[float, float, float] = MISSING
    fingertip_offset_by_body: dict[str, tuple[float, float, float]] = MISSING
    raycast_link_exprs: tuple[str, ...] = MISSING
    base_quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    # Tight box around the hand at the reset pose, env-local, before the margin. Constant
    # because the base pose and the reset joint angles are both fixed. Printed by
    # .output/047-play2perfect-assembly/src/render_hand_box.py.
    hand_box_local: tuple[tuple[float, float, float], tuple[float, float, float]] = MISSING
    arm_armature: dict[str, float] | None = None
    arm_friction: dict[str, float] | None = None
    hand_friction: dict[str, float] | None = None
    hand_dynamic_friction: dict[str, float] | None = None
    hand_effort_limit: dict[str, float] | None = None


_FR3_XHAND_HAND_JOINTS = (
    "thumb_joint0", "thumb_joint1", "thumb_joint2",
    "index_joint0", "index_joint1", "index_joint2",
    "middle_joint0", "middle_joint1",
    "ring_joint0", "ring_joint1",
    "pinky_joint0", "pinky_joint1",
)

_FR3_XHAND_FINGERTIP_BODIES = (
    "index_rota_link2", "mid_link2", "ring_link2",
    "thumb_rota_link2", "pinky_link2",
)

_SHARPA_HAND_JOINTS = (
    "left_1_thumb_CMC_FE", "left_thumb_CMC_AA", "left_thumb_MCP_FE",
    "left_thumb_MCP_AA", "left_thumb_IP",
    "left_2_index_MCP_FE", "left_index_MCP_AA", "left_index_PIP", "left_index_DIP",
    "left_3_middle_MCP_FE", "left_middle_MCP_AA", "left_middle_PIP", "left_middle_DIP",
    "left_4_ring_MCP_FE", "left_ring_MCP_AA", "left_ring_PIP", "left_ring_DIP",
    "left_5_pinky_CMC", "left_pinky_MCP_FE", "left_pinky_MCP_AA",
    "left_pinky_PIP", "left_pinky_DIP",
)

_SHARPA_FINGERTIP_BODIES = (
    "left_index_DP", "left_middle_DP", "left_ring_DP",
    "left_thumb_DP", "left_pinky_DP",
)

# The real XHand, identified 29-30 Sep 2026 at the deploy settings kp 100, kd 0, tor_max 400
# (deployment/fr3_xhand/sysid/HAND_MODEL_20260930.md). Friction is an effort in Nm in Isaac Sim 5.
# stiffness, damping, static friction, dynamic friction, armature, effort limit
_HAND_SYSID = {
    "thumb_joint0": (1.56, 0.093, 0.027, 0.027, 0.0, 0.86),
    "thumb_joint1": (1.58, 0.067, 0.018, 0.0095, 0.0, 0.91),
    "thumb_joint2": (0.32, 0.012, 0.0038, 0.0038, 5e-4, 0.175),
    "index_joint0": (0.31, 0.015, 0.0072, 0.0072, 0.0, 0.18),
    "index_joint1": (1.28, 0.070, 0.014, 0.0088, 0.0, 0.77),
    "index_joint2": (0.28, 0.010, 0.0028, 0.0028, 5e-4, 0.165),
    "middle_joint0": (1.93, 0.124, 0.023, 0.012, 1e-4, 0.93),
    "middle_joint1": (0.30, 0.009, 0.0041, 0.0041, 5e-4, 0.196),
    "ring_joint0": (1.39, 0.082, 0.017, 0.012, 1.5e-3, 0.79),
    "ring_joint1": (0.32, 0.010, 0.0038, 0.0036, 5e-4, 0.205),
    "pinky_joint0": (1.61, 0.099, 0.017, 0.011, 0.0, 0.85),
    "pinky_joint1": (0.31, 0.005, 0.023, 0.023, 5e-4, 0.148),
}

ROBOT_PROFILES: dict[str, RobotCfg] = {
    "kuka-sharpa": RobotCfg(
        urdf="assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf",
        arm_joint_regex="iiwa14_joint_.*",
        hand_joint_regex="left_.*",
        joint_order=(
            "iiwa14_joint_1", "iiwa14_joint_2", "iiwa14_joint_3", "iiwa14_joint_4",
            "iiwa14_joint_5", "iiwa14_joint_6", "iiwa14_joint_7",
        ) + _SHARPA_HAND_JOINTS,
        palm_body="iiwa14_link_7",
        fingertip_body_regex="left_(index|middle|ring|thumb|pinky)_DP",
        fingertip_bodies=_SHARPA_FINGERTIP_BODIES,
        arm_stiffness={
            "iiwa14_joint_1": 600.0, "iiwa14_joint_2": 600.0, "iiwa14_joint_3": 500.0,
            "iiwa14_joint_4": 400.0, "iiwa14_joint_5": 200.0, "iiwa14_joint_6": 200.0,
            "iiwa14_joint_7": 200.0,
        },
        arm_damping={
            "iiwa14_joint_1": 27.027026473513512, "iiwa14_joint_2": 27.027026473513512,
            "iiwa14_joint_3": 24.672186769721083, "iiwa14_joint_4": 22.067474708266914,
            "iiwa14_joint_5": 9.752538131173853, "iiwa14_joint_6": 9.147747263670984,
            "iiwa14_joint_7": 9.147747263670984,
        },
        hand_stiffness={
            "left_1_thumb_CMC_FE": 6.95, "left_thumb_CMC_AA": 13.2, "left_thumb_MCP_FE": 4.76,
            "left_thumb_MCP_AA": 6.62, "left_thumb_IP": 0.9,
            "left_2_index_MCP_FE": 4.76, "left_index_MCP_AA": 6.62,
            "left_index_PIP": 0.9, "left_index_DIP": 0.9,
            "left_3_middle_MCP_FE": 4.76, "left_middle_MCP_AA": 6.62,
            "left_middle_PIP": 0.9, "left_middle_DIP": 0.9,
            "left_4_ring_MCP_FE": 4.76, "left_ring_MCP_AA": 6.62,
            "left_ring_PIP": 0.9, "left_ring_DIP": 0.9,
            "left_5_pinky_CMC": 1.38, "left_pinky_MCP_FE": 4.76, "left_pinky_MCP_AA": 6.62,
            "left_pinky_PIP": 0.9, "left_pinky_DIP": 0.9,
        },
        hand_damping={
            "left_1_thumb_CMC_FE": 0.28676845, "left_thumb_CMC_AA": 0.40845109,
            "left_thumb_MCP_FE": 0.20394083, "left_thumb_MCP_AA": 0.24044435,
            "left_thumb_IP": 0.04190723,
            "left_2_index_MCP_FE": 0.20859232, "left_index_MCP_AA": 0.24595532,
            "left_index_PIP": 0.04243185, "left_index_DIP": 0.03504461,
            "left_3_middle_MCP_FE": 0.2085923, "left_middle_MCP_AA": 0.24595532,
            "left_middle_PIP": 0.04243185, "left_middle_DIP": 0.03504461,
            "left_4_ring_MCP_FE": 0.20859226, "left_ring_MCP_AA": 0.24595528,
            "left_ring_PIP": 0.04243183, "left_ring_DIP": 0.0350446,
            "left_5_pinky_CMC": 0.02782345, "left_pinky_MCP_FE": 0.20859229,
            "left_pinky_MCP_AA": 0.24595528, "left_pinky_PIP": 0.04243183,
            "left_pinky_DIP": 0.0350446,
        },
        hand_armature={
            "left_1_thumb_CMC_FE": 0.0032, "left_thumb_CMC_AA": 0.0032,
            "left_thumb_MCP_FE": 0.00265, "left_thumb_MCP_AA": 0.00265, "left_thumb_IP": 0.0006,
            "left_2_index_MCP_FE": 0.00265, "left_index_MCP_AA": 0.00265,
            "left_index_PIP": 0.0006, "left_index_DIP": 0.00042,
            "left_3_middle_MCP_FE": 0.00265, "left_middle_MCP_AA": 0.00265,
            "left_middle_PIP": 0.0006, "left_middle_DIP": 0.00042,
            "left_4_ring_MCP_FE": 0.00265, "left_ring_MCP_AA": 0.00265,
            "left_ring_PIP": 0.0006, "left_ring_DIP": 0.00042,
            "left_5_pinky_CMC": 0.00012, "left_pinky_MCP_FE": 0.00265,
            "left_pinky_MCP_AA": 0.00265, "left_pinky_PIP": 0.0006, "left_pinky_DIP": 0.00042,
        },
        arm_default_pos={
            "iiwa14_joint_1": -1.571, "iiwa14_joint_2": 1.571, "iiwa14_joint_3": 0.0,
            "iiwa14_joint_4": 1.376, "iiwa14_joint_5": 0.0, "iiwa14_joint_6": 1.485,
            "iiwa14_joint_7": 1.308,
        },
        palm_center_offset=(-0.0, -0.02, 0.16),
        fingertip_offset_by_body={b: (0.02, 0.002, 0.0) for b in _SHARPA_FINGERTIP_BODIES},
        raycast_link_exprs=(
            "/World/envs/env_.*/Robot/iiwa14_link_.*/visuals",
            "/World/envs/env_.*/Robot/left_.*/visuals",
        ),
    ),
    "fr3-xhand": RobotCfg(
        urdf="assets/urdf/fr3_xhand_description/fr3_xhand/fr3_xhand.urdf",
        arm_joint_regex="fr3_joint.*",
        hand_joint_regex="(thumb|index|middle|ring|pinky)_joint.*",
        joint_order=(
            "fr3_joint1", "fr3_joint2", "fr3_joint3", "fr3_joint4",
            "fr3_joint5", "fr3_joint6", "fr3_joint7",
        ) + _FR3_XHAND_HAND_JOINTS,
        palm_body="fr3_link7",
        fingertip_body_regex="(thumb_rota_link2|index_rota_link2|mid_link2|ring_link2|pinky_link2)",
        fingertip_bodies=_FR3_XHAND_FINGERTIP_BODIES,
        arm_stiffness={
            "fr3_joint1": 400.0, "fr3_joint2": 400.0, "fr3_joint3": 400.0,
            "fr3_joint4": 400.0, "fr3_joint5": 400.0, "fr3_joint6": 400.0,
            "fr3_joint7": 400.0,
        },
        arm_damping={
            "fr3_joint1": 31.1307, "fr3_joint2": 31.1307, "fr3_joint3": 27.2029,
            "fr3_joint4": 27.2029, "fr3_joint5": 18.1328, "fr3_joint6": 18.1328,
            "fr3_joint7": 18.1328,
        },
        arm_armature={
            "fr3_joint1": 0.6057, "fr3_joint2": 0.6057, "fr3_joint3": 0.4625,
            "fr3_joint4": 0.4625, "fr3_joint5": 0.2055, "fr3_joint6": 0.2055,
            "fr3_joint7": 0.2055,
        },
        # https://github.com/frankarobotics/franka_description/blob/main/robots/fr3v2/dynamics.yaml
        arm_friction={
            "fr3_joint1": 0.2, "fr3_joint2": 0.2, "fr3_joint3": 0.2,
            "fr3_joint4": 0.2, "fr3_joint5": 0.2, "fr3_joint6": 0.2,
            "fr3_joint7": 0.2,
        },
        hand_stiffness={j: 3.0 for j in _FR3_XHAND_HAND_JOINTS},
        hand_damping={j: 0.1 for j in _FR3_XHAND_HAND_JOINTS},
        hand_armature={j: 0.0 for j in _FR3_XHAND_HAND_JOINTS},
        # Base yawed -90 deg about z so the robot faces the table instead of
        # spending joint 1's range reaching sideways at it.
        base_quat_wxyz=(0.7071067811865476, 0.0, 0.0, -0.7071067811865476),
        hand_box_local=((-0.1589, -0.3137, 0.7749), (0.1589, -0.1040, 0.8388)),
        arm_default_pos={
            "fr3_joint1": 0.0, "fr3_joint2": 0.2618, "fr3_joint3": 0.0,
            "fr3_joint4": -2.0071, "fr3_joint5": 0.0, "fr3_joint6": 2.3562,
            "fr3_joint7": -0.7854,
        },
        palm_center_offset=(0.0, 0.0, 0.16),
        # The centroid of that fingertip's 120 vendor taxels, not the vendor tip point. The tip
        # sits 12.5 mm past the pad centroid on a finger and 15.5 mm on the thumb, and it is off
        # the pad face, so every distance measured from it rewards a tip pose over a pad grasp.
        # T16 centroid (4.874, 0, 30.7745) mm, T30 centroid (0, 35.083167, 3.4695) mm.
        fingertip_offset_by_body={
            "thumb_rota_link2": (0.0, 0.035083167, 0.0034695),
            "index_rota_link2": (0.004874, 0.0, 0.0307745),
            "mid_link2": (0.004874, 0.0, 0.0307745),
            "ring_link2": (0.004874, 0.0, 0.0307745),
            "pinky_link2": (0.004874, 0.0, 0.0307745),
        },
        raycast_link_exprs=(
            "/World/envs/env_.*/Robot/fr3_link.*/visuals",
            "/World/envs/env_.*/Robot/palm/visuals",
            "/World/envs/env_.*/Robot/thumb_.*/visuals",
            "/World/envs/env_.*/Robot/index_.*/visuals",
            "/World/envs/env_.*/Robot/mid_.*/visuals",
            "/World/envs/env_.*/Robot/ring_.*/visuals",
            "/World/envs/env_.*/Robot/pinky_.*/visuals",
        ),
    ),
}

# The fr3-xhand arm (a4) with the identified hand.
ROBOT_PROFILES["fr3-xhand-a4hs"] = ROBOT_PROFILES["fr3-xhand"].replace(
    hand_stiffness={j: v[0] for j, v in _HAND_SYSID.items()},
    hand_damping={j: v[1] for j, v in _HAND_SYSID.items()},
    hand_friction={j: v[2] for j, v in _HAND_SYSID.items()},
    hand_dynamic_friction={j: v[3] for j, v in _HAND_SYSID.items()},
    hand_armature={j: v[4] for j, v in _HAND_SYSID.items()},
    hand_effort_limit={j: v[5] for j, v in _HAND_SYSID.items()},
)
