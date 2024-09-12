import mo_gymnasium as mo_gym
import numpy as np
import random
from morl_baselines.multi_policy.pareto_q_learning.pql import PQL
import os
import wandb

SEEDS = [42]  # 10 seeds
env = mo_gym.make("deep-sea-treasure-concave-v0")
ref_point = np.array([0, -25])

#wandb.init(mode="offline",project="Research Project Logs")
for seed in SEEDS:
    
    print(f"Running experiment with seed {seed}")
    env.reset(seed=seed)
    random.seed(seed)
    np.random.seed(seed)
    env.reset(seed=seed)
    
    agent = PQL(
        env,
        ref_point,
        gamma=0.9,
        initial_epsilon=1,
        epsilon_decay_steps=10000,
        final_epsilon=0.1,
        seed=seed,
        experiment_name="Pareto Q-Learning in DST",
        project_name="Research Project Logs",
        log=True,)

    pf = agent.train(
        total_timesteps=1000,
        log_every=100,
        action_eval="hypervolume",
        known_pareto_front=env.pareto_front(gamma=0.9),
        ref_point=ref_point,
        eval_env=env,)
    print(pf)
    agent.close_wandb()

    # Execute a policy
    target = np.array(pf.pop())
    print(f"Tracking {target}")
    reward = agent.track_policy(target, env=env)
    print(f"Obtained {reward}")
    #wandb.finish()