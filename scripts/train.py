import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.distributions.normal import Normal

from sim.launch_app import launch_app
from utils.config import load_config, config

sim_app, args_cli = launch_app()
load_config(r"configs\yaml\train.yaml")

import isaaclab.sim as sim_utils
from isaaclab.controllers import OperationalSpaceController
from isaaclab.scene import InteractiveScene
from isaaclab.assets import Articulation
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.sim import SimulationContext, SimulationCfg

from configs.python.scene_cfg import SceneCfg
from configs.python.env_cfg import EnvironmentCfg
from sim.environment import Environment

from rl.ppo import Actor, Critic
from rl.rollout import Rollout


def main() -> None:
    """
    Main function ran on file execution
    """
    # Create the environment
    env_cfg: EnvironmentCfg = EnvironmentCfg()
    env: Environment = Environment(env_cfg,)
    # Load the simulation
    sim: sim_utils.SimulationContext = sim_utils.SimulationContext(env_cfg.sim,)
    sim.reset()
    # Extract the scene and robot
    scene: InteractiveScene = env.scene
    robot: Articulation = scene["robot"]
    robot.update(dt=sim_dt,)
    sim_dt: float = sim.get_physics_dt()
    
    # Create RL objects
    policy: Actor = Actor()
    value: Critic = Critic()
    
    optimizer: Adam = Adam([
        {"params": policy.parameters(), "lr": config["rl"]["policy_lr"]},
        {"params": value.parameters(), "lr": config["rl"]["value_lr"]}
    ])
    
    """ Training Loop """
    # Epoch = num rollouts collected
    for epoch in range(config["rl"]["epochs"]):
        # Rollout collection phase
        obs: torch.Tensor = env.reset()
        rollout: Rollout = Rollout(initial_state=obs,)
        # Loop until rollout is at capacity
        while len(rollout) < config["rl"]["rollout_length"]:
            # Sample action from policy
            mean, variance = policy(obs,)
            action: torch.Tensor = Normal(mean, variance,).sample()
            # Step in the environment
            obs, rew, term, trunc, _, = env.step(action,)
            # Add to rollout
            rollout.add(
                obs, action, mean, variance, rew, value(obs,), term or trunc,
            )
        
        # Batch rollout
        dataloader: DataLoader = DataLoader(
            dataset=rollout,
            batch_size=config["rl"]["batch_size"],
            shuffle=True,
        )
        # Training phase
        for _ in range(config["rl"]["iterations"]):
            rews_h, dones_h, value_outs_h = rollout.get_horizon()
            # Compute advantages
            advantages: torch.Tensor = policy.gae(
                rewards=rews_h,
                dones=dones_h,
                value_outs=value_outs_h,
            )
            
            for iteration in range(config["rl"]["iterations"]):
                for obs, actions, old_means, old_variances, rews, old_value_outs in dataloader:
                    optimizer.zero_grad()
                    # Compute advantages and loss
                    value_outs: torch.Tensor = value(obs,)
                    means, variances = policy(obs,)
                    policy_dist: Normal = Normal(means, variances)
                    
                    policy_loss: torch.Tensor = policy.policy_objective(
                        policy_dist=policy_dist,
                        old_policy_dist=Normal(old_means, old_variances),
                        actions=actions,
                        advantages=advantages,
                    )
                    value_loss: torch.Tensor = value.value_objective(
                        value_outs=value_outs,
                        old_value_outs=old_value_outs,
                        advantages=advantages,
                    )
                    
                    # Backpropagate
                    loss: torch.Tensor = policy_loss + config["rl"]["value_coef"] * value_loss + config["rl"]["entropy_coef"] * policy_dist.entropy()
                    loss.backward()
                    optimizer.step()
        

if __name__ == "__main__":
    main()