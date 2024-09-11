import mo_gymnasium as mo_gym
import numpy as np
import random
from morl_baselines.multi_policy.pareto_q_learning.pql import PQL
import os

SEEDS = [42]  # 10 seeds
env = mo_gym.make("deep-sea-treasure-concave-v0")
ref_point = np.array([0, -25])
os.environ['WANDB_MODE'] = 'offline'
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
        initial_epsilon=0.1,
        final_epsilon=0.1,
        seed=seed,
        project_name="Research Project Logs",
        experiment_name="Pareto Q-Learning in DST",
        log=True,)

    pf = agent.train(
        total_timesteps=1000000,
        log_every=100,
        action_eval="hypervolume",
        known_pareto_front=env.pareto_front(gamma=0.99),
        ref_point=ref_point,
        eval_env=env,)
    print(pf)

    # Execute a policy
    target = np.array(pf.pop())
    print(f"Tracking {target}")
    reward = agent.track_policy(target, env=env)
    print(f"Obtained {reward}")