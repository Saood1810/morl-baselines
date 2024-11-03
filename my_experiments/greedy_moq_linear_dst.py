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

# Set seeds for reproducibility
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51,52,53]  # 10 seeds

# Environment instantiation
env = MORecordEpisodeStatistics(mo_gym.make("deep-sea-treasure-concave-v0"), gamma=0.9)
eval_env =MORecordEpisodeStatistics(mo_gym.make("deep-sea-treasure-concave-v0"), gamma=0.9)


for seed in SEEDS:
  print(f"Running experiment with seed {seed}")
  exp_name = f"Greedy Linear Experiment with seed {seed}"
  rows, cols = 11, 400   #11 configurations trained across 400 iterations
  random.seed(seed)
  np.random.seed(seed)
  env.reset(seed=seed)
  eval_env.reset(seed=seed)

  moq_eval_rewards=np.zeros((rows,cols,2)) #Array to store the rewards for each weight combination across 400 iterations
  for i in range(0, 11):
      print(i)
      
          
      weights = np.array([1 - (i / 10), i / 10])

    #This experiments focuses on performance with minimal exploration , thus we decay the epsilon value faster
      linear = MOQLearning(env, scalarization=weighted_sum,initial_epsilon=1,final_epsilon=0.1,epsilon_decay_steps=0.01*400000, gamma=0.9, weights=weights,seed=seed, log=False)

    #For each weight configuration, we train it for 400 iterations
      for z in range(0, 400):
          linear.train(
              total_timesteps=1000,
              reset_num_timesteps= False,
              start_time=time.time(),
              eval_env=eval_env,
          )
          #eval_env.reset()
          
          #we evaluate the policy after every 1000 timesteps
          _,_,_,disc_reward=(eval_mo(linear, env=eval_env, w=weights))
          moq_eval_rewards[i][z]=disc_reward
  eval_env.reset(seed=seed)
  
  #Evaluate the performance of the MOQ agent across all weight configurations by calculating the pf approximation .Hypervolume, Cardinality, IGD and Sparsity
  pf,hypervolume_scores,cardinality_scores,igd_scores,sparsity_scores=evaluate(moq_eval_rewards,np.array([0,-50]),eval_env)
  #Log the results to wandb
  log_results(pf,hypervolume_scores,cardinality_scores,igd_scores,sparsity_scores,"Research Project Logs",exp_name,"Greedy MOQ Linear DST")
  print("Greedy MOQ Linear Results for seed: ",seed)
  print(pf)
  
