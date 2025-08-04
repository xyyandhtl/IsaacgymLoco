import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
from pathlib import Path

MOTION_FILES_DIR = str(Path(__file__).parent)
VISUALIZE_RETARGETING = True

URDF_FILENAME = f"{LEGGED_GYM_ROOT_DIR}/resources/robots/aliengo/urdf/aliengo.urdf"

OUTPUT_DIR = f"{MOTION_FILES_DIR}/mocap_motions_aliengo"

REF_POS_SCALE = 0.825
INIT_POS = np.array([0, 0, 0.4])
INIT_ROT = np.array([0, 0, 0, 1.0])

SIM_TOE_JOINT_IDS = [8, 15, 22, 29]
SIM_HIP_JOINT_IDS = [2, 9, 16, 23]
SIM_ROOT_OFFSET = np.array([0, 0, -0.04])
SIM_TOE_OFFSET_LOCAL = [
    np.array([0, -0.05, 0.0]),
    np.array([0, 0.05, 0.0]),
    np.array([0, -0.05, 0.0]),
    np.array([0, 0.05, 0.0])
]
TOE_HEIGHT_OFFSET = 0.02

DEFAULT_JOINT_POSE = np.array([0, 0.8, -1.5, 0, 0.8, -1.5, 0, 0.8, -1.5, 0, 0.8, -1.5])
# DEFAULT_JOINT_POSE = np.array([0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8])
JOINT_DAMPING = [0.1, 0.05, 0.01,
                 0.1, 0.05, 0.01,
                 0.1, 0.05, 0.01,
                 0.1, 0.05, 0.01]

FORWARD_DIR_OFFSET = np.array([0, 0, 0])

FR_FOOT_NAME = "FR_foot"
FL_FOOT_NAME = "FL_foot"
HR_FOOT_NAME = "RR_foot"
HL_FOOT_NAME = "RL_foot"

MOCAP_MOTIONS = [
    # Output motion name, input file, frame start, frame end, motion weight.
    [
        "pace0",
        f"{MOTION_FILES_DIR}/keypoint_datasets/ai4animation/dog_walk00_joint_pos.txt", 162, 201, 1
    ],
    [
        "pace1",
        f"{MOTION_FILES_DIR}/keypoint_datasets/ai4animation/dog_walk00_joint_pos.txt", 201, 400, 1
    ],
    [
        "pace2",
        f"{MOTION_FILES_DIR}/keypoint_datasets/ai4animation/dog_walk00_joint_pos.txt", 400, 600, 1
    ],
    [
        "trot0",
        f"{MOTION_FILES_DIR}/keypoint_datasets/ai4animation/dog_walk03_joint_pos.txt", 448, 481, 1
    ],
    [
        "trot1",
        f"{MOTION_FILES_DIR}/keypoint_datasets/ai4animation/dog_walk03_joint_pos.txt", 400, 600, 1
    ],
    [
        "trot2",
        f"{MOTION_FILES_DIR}/keypoint_datasets/ai4animation/dog_run04_joint_pos.txt", 480, 663, 1
    ],
    [
        "canter0",
        f"{MOTION_FILES_DIR}/keypoint_datasets/ai4animation/dog_run00_joint_pos.txt", 430, 480, 1
    ],
    [
        "canter1",
        f"{MOTION_FILES_DIR}/keypoint_datasets/ai4animation/dog_run00_joint_pos.txt", 380, 430, 1
    ],
    [
        "canter2",
        f"{MOTION_FILES_DIR}/keypoint_datasets/ai4animation/dog_run00_joint_pos.txt", 480, 566, 1
    ],
    [
        "right_turn0",
        f"{MOTION_FILES_DIR}/keypoint_datasets/ai4animation/dog_walk09_joint_pos.txt", 1085, 1124, 1.5
    ],
    [
        "right_turn1",
        f"{MOTION_FILES_DIR}/keypoint_datasets/ai4animation/dog_walk09_joint_pos.txt", 560, 670, 1.5
    ],
    [
        "left_turn0",
        f"{MOTION_FILES_DIR}/keypoint_datasets/ai4animation/dog_walk09_joint_pos.txt", 2404, 2450, 1.5
    ],
    [
        "left_turn1",
        f"{MOTION_FILES_DIR}/keypoint_datasets/ai4animation/dog_walk09_joint_pos.txt", 120, 220, 1.5
    ]
]