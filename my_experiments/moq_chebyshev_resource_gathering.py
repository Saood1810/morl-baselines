from utilities import log_results,evaluate,generate_combinations
import time
import gymnasium as gym
import mo_gymnasium as mo_gym
import numpy as np
import wandb
from mo_gymnasium.utils import MORecordEpisodeStatistics

from morl_baselines.common.evaluation import eval_mo
from morl_baselines.common.evaluation import policy_evaluation_mo
from morl_baselines.common.scalarization import tchebicheff
from morl_baselines.single_policy.ser.mo_q_learning import MOQLearning

from morl_baselines.multi_policy.pareto_q_learning.pql import PQL

from morl_baselines.common.evaluation import policy_evaluation_mo
from morl_baselines.common.evaluation import log_all_multi_policy_metrics
from morl_baselines.common.pareto import filter_pareto_dominated
from morl_baselines.common.performance_indicators import hypervolume
import matplotlib.pyplot as plt
from morl_baselines.common.performance_indicators import cardinality
from morl_baselines.common.performance_indicators import igd
from morl_baselines.common.performance_indicators import sparsity
from morl_baselines.common.scalarization import weighted_sum


import numpy as np
import time

from morl_baselines.common.evaluation import policy_evaluation_mo

# Run the experiment for multiple seeds
import random


SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]  # 10 seeds
env = MORecordEpisodeStatistics(mo_gym.make("resource-gathering-v0"), gamma=0.9)
eval_env = MORecordEpisodeStatistics(mo_gym.make("resource-gathering-v0"), gamma=0.9)
for seed in SEEDS:
  
  weight_combinations = generate_combinations()
  print(f"Running experiment with seed {seed}")
  exp_name = f"Chebyshev in RG Experiment with gamma 0.9 and seed {seed}"
  rows, cols = len(weight_combinations), 400  
  random.seed(seed)
  np.random.seed(seed)
  env.reset(seed=seed)
  eval_env.reset(seed=seed)

  
 
  moq_eval_rewards=np.zeros((rows,cols,3))

  for i in range(0, len(weight_combinations)):
    print(i)
    env.reset(seed=seed)
    eval_env.reset(seed=seed)
    scalarization = tchebicheff(tau=6.0, reward_dim=3)
    weights =np.array(weight_combinations[i])

    agent = MOQLearning(env, scalarization=scalarization,initial_epsilon=1,final_epsilon=0.1,epsilon_decay_steps=0.01*400000, gamma=0.9, weights=weights, log=False)

    for z in range(0, 400):
        agent.train(
            total_timesteps=1000,
            reset_num_timesteps= False,
            start_time=time.time(),
            eval_env=eval_env,
        )
        #eval_env.reset()
        _,_,_,disc_reward=(eval_mo(agent, env=eval_env, w=weights))
        moq_eval_rewards[i][z]=disc_reward
        
  
  pf,hypervolume_scores,cardinality_scores,igd_scores,sparsity_scores=evaluate(moq_eval_rewards,np.array([-1,-1,-2]),eval_env,0.90)
  log_results(pf,hypervolume_scores,cardinality_scores,igd_scores,sparsity_scores,"Research Project Logs V6",exp_name,"Chebyshev Resource Gathering gamma 0.9")
  print("Balanced MOQ Chebyshev Results for seed: ",seed)
  print(pf)
 
  
