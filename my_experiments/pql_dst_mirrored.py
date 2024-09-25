import mo_gymnasium as mo_gym
import numpy as np
import random
from mo_gymnasium.utils import MORecordEpisodeStatistics

from morl_baselines.multi_policy.pareto_q_learning.pql import PQL
import os
import wandb
from utilities import eval_pql,log_results

SEEDS = [42,43,44,45,46,47,48,49,50,51]  # 10 seeds
env = MORecordEpisodeStatistics(mo_gym.make("deep-sea-treasure-mirrored-v0"), gamma=0.9)
eval_env = MORecordEpisodeStatistics(mo_gym.make("deep-sea-treasure-mirrored-v0"), gamma=0.9)
ref_point = np.array([0, -50])


#wandb.init(mode="offline",project="Research Project Logs")
for seed in SEEDS:
    wandb.init(mode="offline",project="Research Project Logs V6",group=" Pareto Q-Learning 800k 0.01 in DST Mirrored V2",name="Pareto Q-Learning 0.01 800k in DST Mirrored with seed "+str(seed))
    
    print(f"Running experiment with seed {seed}")
   
    random.seed(seed)
    np.random.seed(seed)
    env.reset(seed=seed)
    eval_env.reset(seed=seed)
    
    agent = PQL(
        env,
        ref_point,
        gamma=0.9,
        initial_epsilon=1,
        epsilon_decay_steps=0.01*800000,
        final_epsilon=0.1,
        seed=seed,
        experiment_name="Pareto Q-Learning 800k 0.01 in DST mirrored with seed "+str(seed),
        project_name="Research Project Logs V6",
        log=True,)

    pf = agent.train(
        total_timesteps=800000,
        log_every=1000,
        action_eval="hypervolume",
        known_pareto_front=env.pareto_front(gamma=0.9),
        ref_point=ref_point,
        eval_env=eval_env,)
    
    print(pf)
    
    wandb.finish()
    #pf_approx,hypervolume_scores,cardinality_scores,igd_scores,sparsity_scores=eval_pql(policies,ref_point,env,agent.gamma)
    #log_results(pf_approx,hypervolume_scores,cardinality_scores,igd_scores,sparsity_scores,"Research Project Logs","Pareto Q-Learning","Pareto Q-Learning")
    
    
    #agent.close_wandb()

    