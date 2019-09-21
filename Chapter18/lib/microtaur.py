import pybullet as p
from pybullet_envs import robot_bases, env_bases, scene_stadium
import re
import os
import gym
from gym.envs.registration import register as gym_register
import enum
import logging
import tempfile
import numpy as np
from typing import List, Optional, Tuple

log = logging.getLogger(__name__)


ENV_ID = "Microtaur-v1"


class FourShortLegsRobot(robot_bases.MJCFBasedRobot):
    IMU_JOINT_NAME = "IMUroot"
    SERVO_JOINT_NAMES = ('servo_rb', 'servo_rf',
                         'servo_lb', 'servo_lf')

    def __init__(self, time_step: float, zero_yaw: bool):
        action_dim = 4
        # current servo positions + pitch, roll and yaw angles
        obs_dim = 4 + 3

        super(FourShortLegsRobot, self).__init__(
            "", "four_short_legs", action_dim, obs_dim)
        self.time_step = time_step
        self.imu_link_idx = None
        self.scene = None
        self.state_id = None
        self.pose = None
        self.zero_yaw = zero_yaw

    def close(self):
        pass

    def get_model_dir(self):
        return "models"

    def get_model_file(self):
        return "four_short_legs.xml"

    # override reset to load model from our path
    def reset(self, bullet_client):
        self._p = bullet_client
        if not self.doneLoading:
            self._p.setAdditionalSearchPath(self.get_model_dir())
            self.objects = self._p.loadMJCF(self.get_model_file())
            assert len(self.objects) == 1
            self.parts, self.jdict, \
            self.ordered_joints, self.robot_body = \
                self.addToScene(self._p, self.objects)

            self.imu_link_idx = self._get_imu_link_index(
                self.IMU_JOINT_NAME)
            self.doneLoading = 1
            self.state_id = self._p.saveState()
        else:
            self._p.restoreState(self.state_id)
        self.robot_specific_reset(self._p)
        self.pose = self.robot_body.pose()
        return self.calc_state()

    def _get_imu_link_index(self, joint_name):
        for j_idx in range(self._p.getNumJoints(self.objects[0])):
            info = self._p.getJointInfo(self.objects[0], j_idx)
            name = str(info[1], encoding='utf-8')
            if name == joint_name:
                return j_idx
        raise RuntimeError

    def _joint_name_direction(self, j_name):
        # forward legs are rotating in inverse direction
        if j_name[-1] == 'f':
            return -1
        else:
            return 1

    def show_joints(self):
        for j_idx in range(self._p.getNumJoints(self.objects[0])):
            info = self._p.getJointInfo(self.objects[0], j_idx)
            print("Info for joint %d" % j_idx)
            print("  name=%s, type=%d, qIndex=%d, uIndex=%d" % (
                str(info[1], encoding='utf-8'), info[2], info[3], info[4]))
            print("  lowLimit=%.2f, highLimit=%.2f, axis=%s" % (
                info[8], info[9], info[13]
            ))
            print("  parentPos=%s, parentOrient=%s" % (
                info[14], info[15]))
            print("  parentIndex=%d" % (info[16],))

            state = p.getJointState(1, j_idx)
            print("  pos=%.2f, vel=%.2f, torque=%.2f" % (
                state[0], state[1], state[3]
            ))
            print("")

    def get_link_pos(self, link_id=None):
        if link_id is None:
            link_id = self.imu_link_idx
        return self._p.getLinkState(self.objects[0], link_id)[0]

    def get_link_orient(self, link_id=None):
        if link_id is None:
            link_id = self.imu_link_idx
        res = self._p.getLinkState(self.objects[0], link_id)[1]
        return res

    def get_link_lin_vel(self, link_id=None):
        if link_id is None:
            link_id = self.imu_link_idx
        return self._p.getLinkState(self.objects[0], link_id, 1)[6]

    def get_link_ang_vel(self, link_id=None):
        if link_id is None:
            link_id = self.imu_link_idx
        return self._p.getLinkState(self.objects[0], link_id, 1)[7]

    def get_joint_torque(self, name):
        j = self.jdict[name]
        res = j._p.getJointState(j.bodies[j.bodyIndex], j.jointIndex)
        return res

    def enable_joint_force_sensor(self, name, enable=True):
        j = self.jdict[name]
        j._p.enableJointForceTorqueSensor(j.bodies[j.bodyIndex], j.jointIndex, enable)

    def calc_state(self):
        res = []
        for idx, j_name in enumerate(self.SERVO_JOINT_NAMES):
            j = self.jdict[j_name]
            dir = self._joint_name_direction(j_name)
            res.append(j.get_position() * dir / np.pi)
        rpy = self.pose.rpy()
        if self.zero_yaw:
            res.extend(rpy[:2])
            res.append(0.0)
        else:
            res.extend(rpy)
        return np.array(res, copy=False)

    def robot_specific_reset(self, client):
        for j in self.ordered_joints:
            j.reset_current_position(0, 0)

    def apply_action(self, action):
        for j_name, act in zip(self.SERVO_JOINT_NAMES, action):
            pos_mul = self._joint_name_direction(j_name)
            j = self.jdict[j_name]
            res_act = pos_mul * act * np.pi
            self._p.setJointMotorControl2(
                j.bodies[j.bodyIndex], j.jointIndex,
                controlMode=p.POSITION_CONTROL,
                targetPosition=res_act, targetVelocity=50,  # tune
                positionGain=1, velocityGain=1, force=2,
                maxVelocity=100)


class RewardScheme(enum.Enum):
    MoveForward = 0
    Height = 1
    HeightOrient = 2


class FourShortLegsEnv(env_bases.MJCFBaseBulletEnv):
    """
    Actions are servo positions, observations are current positions of servos plus three 3-axis values from
    accelerometer, gyroscope and magnetometer
    """
    HEIGHT_BOUNDARY = 0.035
    ORIENT_TOLERANCE = 1e-2

    def __init__(self, render=False, target_dir=(0, 1),
                 timestep: float = 0.01, frameskip: int = 4,
                 reward_scheme: RewardScheme = RewardScheme.Height,
                 zero_yaw: bool = False):
        self.frameskip = frameskip
        self.timestep = timestep / self.frameskip
        self.reward_scheme = reward_scheme
        robot = FourShortLegsRobot(self.timestep, zero_yaw=zero_yaw)
        super(FourShortLegsEnv, self).__init__(robot, render=render)
        self.target_dir = target_dir
        self.stadium_scene = None
        self._prev_pos = None
        self._cam_dist = 1

    def create_single_player_scene(self, bullet_client):
        self.stadium_scene = scene_stadium.SinglePlayerStadiumScene(
            bullet_client, gravity=9.8, timestep=self.timestep,
            frame_skip=self.frameskip)
        return self.stadium_scene

    def _reward(self):
        result = 0
        if self.reward_scheme == RewardScheme.MoveForward:
            pos = self.robot.get_link_pos()
            if self._prev_pos is None:
                self._prev_pos = pos
                return 0.0
            dx = pos[0] - self._prev_pos[0]
            dy = pos[1] - self._prev_pos[1]
            self._prev_pos = pos
            result = dx * self.target_dir[0] + \
                     dy * self.target_dir[1]
        elif self.reward_scheme == RewardScheme.Height:
            result = int(self._reward_check_height())
        elif self.reward_scheme == RewardScheme.HeightOrient:
            cond = self._reward_check_height() and \
                   self._reward_check_orient()
            result = int(cond)
        return result

    def _reward_check_height(self):
        """
        Check height criteria
        :return: True if z coordinate of IMU unit is righer than boundary
        """
        return self.robot.get_link_pos()[-1] > self.HEIGHT_BOUNDARY

    def _reward_check_orient(self):
        """
        Check orientation criteria
        :return: True if IMU lays in XY plane, in other words, roll and pitch is almost zero
        """
        orient = self.robot.get_link_orient()
        orient = p.getEulerFromQuaternion(orient)
        return (abs(orient[0]) < self.ORIENT_TOLERANCE) and \
               (abs(orient[1]) < self.ORIENT_TOLERANCE)

    def step(self, action):
        self.robot.apply_action(action)
        self.scene.global_step()
        return self.robot.calc_state(), self._reward(), False, {}

    def reset(self):
        r = super(FourShortLegsEnv, self).reset()
        if self.isRender:
            distance, yaw = 0.2, 30
            self._p.resetDebugVisualizerCamera(
                distance, yaw, -20, [0, 0, 0])
        return r

    def close(self):
        self.robot.close()
        super(FourShortLegsEnv, self).close()


class FourShortLegsRobotParametrized(FourShortLegsRobot):
    """
    Parametrized version of basic robot
    """
    DEFAULTS = {
        "servo.targetVelocity": 50,
        "servo.positionGain": 1,
        "servo.velocityGain": 1,
        "servo.force": 20,
        "servo.maxVelocity": 100,
        "compiler.totalMass": 0.05,
        "joint.armature": 0,
        "joint.damping": 1,
        "joint.frictionloss": 0,
        "position.kp": 10,
        "geom.friction": 0.5,
    }

    FILE_PARAMS = {"compiler.totalMass", "joint.armature", "joint.damping", "joint.frictionloss", "position.kp"}
    BOUNDS = {
        "servo.targetVelocity": (0.1, 100),
        "servo.positionGain": (0.1, 10),
        "servo.velocityGain": (0.1, 10),
        "servo.force": (1, 100),
        "servo.maxVelocity": (90, 200),
        "compiler.totalMass": (0.01, 0.1),
        "joint.armature": (0, 1),
        "joint.damping": (0, 1),
        "joint.frictionloss": (0, 1),
        "position.kp": (1, 20),
        "geom.friction": (0.1, 1.0),
    }

    MODEL_TEMPLATE_FILE = "four_short_legs.xml.tmpl"
    REMOVE_TEMP_FILES = True

    def __init__(self, time_step: float, imu_observations: bool, params_vector: List[float], obs_bias: bool = False):
        super(FourShortLegsRobotParametrized, self).__init__(time_step, imu_observations, obs_bias)
        self._params = self.vector_to_params(params_vector)
        self._model_file_name = None

    def get_model_file(self):
        self._model_file_name = self._build_model_file()
        return self._model_file_name

    def close(self):
        if self._model_file_name is not None and self.REMOVE_TEMP_FILES:
            os.unlink(os.path.join(self.get_model_dir(), self._model_file_name))
            self._model_file_name = None

    def _build_model_file(self):
        """
        Create a MuJoCo model file with all values expanded
        :return:
        """
        file = tempfile.NamedTemporaryFile(mode="w+t", dir=self.get_model_dir(), encoding='utf-8', delete=False, suffix=".xml")
        try:
            path = os.path.join(self.get_model_dir(), self.MODEL_TEMPLATE_FILE)
            with open(path, "rt", encoding='utf-8') as fd:
                for l in fd:
                    res_parts = []
                    prev_pos = 0
                    for match in re.finditer(r"{{([\w.]+)}}", l):
                        name = match.group(1)
                        value = self._params[name]
                        res_parts.append(l[prev_pos:match.start()])
                        res_parts.append(str(value))
                        prev_pos = match.end()
                    res_parts.append(l[prev_pos:])
                    file.write("".join(res_parts))
        finally:
            file.close()
        return file.name

    @classmethod
    def params_vector(cls, **kwargs) -> List[float]:
        res = []
        for name in sorted(cls.DEFAULTS.keys()):
            val = kwargs.get(name, cls.DEFAULTS[name])
            res.append(float(val))
        return res

    @classmethod
    def params_bounds(cls) -> List[Tuple[float, float]]:
        res = []
        for name in sorted(cls.BOUNDS.keys()):
            res.append(cls.BOUNDS[name])
        return res

    @classmethod
    def vector_to_params(cls, vector: List[float]) -> dict:
        res = {}
        for name, value in zip(sorted(cls.DEFAULTS.keys()), vector):
            res[name] = value
        return res

    def apply_action(self, action):
        tgt_velocity = self._params["servo.targetVelocity"]
        pos_gain = self._params["servo.positionGain"]
        vel_gain = self._params["servo.velocityGain"]
        force = self._params["servo.force"]
        vel_max = self._params["servo.maxVelocity"]

        for j_name, act in zip(self.SERVO_JOINT_NAMES, action):
            j = self.jdict[j_name]
            pos_mul = self._joint_name_direction(j_name)

            res_action = pos_mul * act * np.pi
            self._p.setJointMotorControl2(
                j.bodies[j.bodyIndex], j.jointIndex, controlMode=p.POSITION_CONTROL,
                targetPosition=res_action, targetVelocity=tgt_velocity,  positionGain=pos_gain,
                velocityGain=vel_gain, force=force, maxVelocity=vel_max)


def _gen_range(start, stop, delta, eps=1e-4):
    pos = start
    stop_loop = False
    while not stop_loop:
        yield pos
        pos += delta
        stop_loop = abs(pos - stop) < eps


def generate_positions(start, steps, min_pos=0, max_pos=1):
    """
    Generate sequence of positions with given count of steps it does full range, for example, for [0, 1] range:
    [0.0, 0.5, 1.0, 0.5, 0.0]
    :param start: position to start from
    :param steps: count of steps to generate for min-max interval
    :return: iterator
    """
    delta = (max_pos - min_pos) / steps
    yield from _gen_range(start, max_pos, delta)
    yield from _gen_range(max_pos, start, -delta)


def register():
    gym_register(ENV_ID, entry_point="lib.microtaur:FourShortLegsEnv", max_episode_steps=1000)
