import mo_gymnasium as mo_gym
import numpy as np
import random
from mo_gymnasium.utils import MORecordEpisodeStatistics

from morl_baselines.multi_policy.pareto_q_learning.pql import PQL
import os
import wandb
from utilities import eval_pql,log_results

SEEDS = [42,43,44,45,46,47,48,49,50,51]  # 10 seeds
#
env = MORecordEpisodeStatistics(mo_gym.make("deep-sea-treasure-mirrored-v0"), gamma=0.9)
eval_env = MORecordEpisodeStatistics(mo_gym.make("deep-sea-treasure-mirrored-v0"), gamma=0.9)
ref_point = np.array([0, -50])


#wandb.init(mode="offline",project="Research Project Logs")
for seed in SEEDS:
    
    #Wandb Initialization
    # Had to use offline mode since CHPC cluster does not have internet access 
    wandb.init(mode="offline",project="Research Project Logs V7",group=" Pareto Q Learning in DST Mirrored UPDATED",name="Pareto Q-Learning in DST Mirrored with seed "+str(seed))
    
    print(f"Running experiment with seed {seed}")
   
    random.seed(seed)
    np.random.seed(seed)
    env.reset(seed=seed)
    eval_env.reset(seed=seed)
    
    #Algorithm Instantiation
    agent = PQL(
        env,
        ref_point,
        gamma=0.9,
        initial_epsilon=1,
        epsilon_decay_steps=800000,
        final_epsilon=0.1,
        seed=seed,
        experiment_name="Pareto Q-Learning 800k 0.01 in DST mirrored with seed "+str(seed),
        project_name="Research Project Logs V7",
        log=True,)

 #Training
    pf = agent.train(
        total_timesteps=800000,
        log_every=1000, #Every 1000 timesteps, log the results
        action_eval="hypervolume",
        known_pareto_front=env.pareto_front(gamma=0.9),
        ref_point=ref_point,
        eval_env=eval_env,)
    
    
    print(pf)
    
    #Plots are automatically logged to wandb
    wandb.finish()
   

    