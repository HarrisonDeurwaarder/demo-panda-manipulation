from isaaclab.envs import DirectRLEnvCfg
from isaaclab.utils import configclass
from isaaclab.sim import SimulationCfg

from utils.config import config
from configs.python.scene_cfg import SceneCfg


@configclass
class EnvironmentCfg(DirectRLEnvCfg):
    """
    RL environment configuration
    """
    # Environment config
    decimation: int = config["env"]["decimation"]
    episode_length: float = config["env"]["episode_length"]
    action_space: int = config["env"]["action_space"]
    obs_space: int = config["env"]["obs_space"]
    state_space: int = config["env"]["state_space"]
    
    # Simulation config
    sim: SimulationCfg = SimulationCfg(
        dt=config["scene"]["dt"],
        render_interval=config["scene"]["render_interval"],
    )
    
    # Scene config
    scene: SceneCfg = SceneCfg(
        num_envs=config["scene"]["num_envs"],
        env_spacing=config["scene"]["env_spacing"],
        replicate_physics=config["scene"]["replicate_physics"],
        clone_in_fabric=config["scene"]["clone_in_fabric"],
    )