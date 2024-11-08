from utilities import log_results,evaluate
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
# Environment Instantiation
env = MORecordEpisodeStatistics(mo_gym.make("deep-sea-treasure-concave-v0"), gamma=0.9)
eval_env = MORecordEpisodeStatistics(mo_gym.make("deep-sea-treasure-concave-v0"), gamma=0.9)

# Run the experiment for multiple seeds
for seed in SEEDS:
  print(f"Running experiment with seed {seed}")
  exp_name = f"Balanced Linear Experiment in DST with seed {seed}"
  rows, cols = 11, 400   #11 Weight combinations, 400 iterations
  random.seed(seed)
  np.random.seed(seed)
  env.reset(seed=seed)
  eval_env.reset(seed=seed)
 
  moq_eval_rewards=np.zeros((rows,cols,2)) #Numpy array to store rewards for the 11 weight configurations across 400 iterations
 
  for i in range(0, 11):
      print(i)
      
         
      weights = np.array([1 - (i / 10), i / 10])# Calculate the weights tuple - (1-w1,w1) where w1 ranges from 0 to 1 in steps of 0.1

    #Algorithm Instantiation with Linear Scalarization and Hyperparameters
      linear = MOQLearning(env, scalarization=weighted_sum,initial_epsilon=1,final_epsilon=0.1,epsilon_decay_steps=0.01*400000, gamma=0.9, weights=weights,seed=seed, log=False)

#Train the agent for 400 iterations
      for z in range(0, 400):
          linear.train(
              total_timesteps=1000,
              reset_num_timesteps= False,
              start_time=time.time(),
              eval_env=eval_env,
          )
          #eval_env.reset()
          #At end of ieteration, evaluate the agent and store the discounted reward
          _,_,_,disc_reward=(eval_mo(linear, env=eval_env, w=weights))
          moq_eval_rewards[i][z]=disc_reward
  eval_env.reset(seed=seed)
  
 #Evaluate the performance of the MOQ agent across all weight configurations by calculating the pf approximation .Hypervolume, Cardinality, IGD and Sparsity
  pf,hypervolume_scores,cardinality_scores,igd_scores,sparsity_scores=evaluate(moq_eval_rewards,np.array([0,-50]),eval_env,0.9)
  
  #log results to wandb
  log_results(pf,hypervolume_scores,cardinality_scores,igd_scores,sparsity_scores,"Research Project Logs V7",exp_name,"MO Q Learning with Linear Scalarization and Low Exploration in DST Concave")
  
  print("Balanced MOQ Linear Results for seed: ",seed)
  print(pf)

  
