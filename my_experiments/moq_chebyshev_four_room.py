from utilities import log_results,evaluate,generate_combinations,eval_unknown,log_unknown_results
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
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]  # 10 seeds

# Environment instantiation
env = MORecordEpisodeStatistics(mo_gym.make("four-room-v0"), gamma=0.99)
eval_env = MORecordEpisodeStatistics(mo_gym.make("four-room-v0"), gamma=0.99)

# Run the experiment for multiple seeds
for seed in SEEDS:
  # Generate weight combinations
  weight_combinations = generate_combinations()
  print(f"Running experiment with seed {seed}")
  exp_name = f"MOQ Chebyshev Experiment in RG with seed {seed}"
  rows, cols = len(weight_combinations), 800  
  random.seed(seed)
  np.random.seed(seed)
  env.reset(seed=seed)
  eval_env.reset(seed=seed)

  
 # Array to store the rewards for each weight combination across 800 iterations
  moq_eval_rewards=np.zeros((rows,cols,3))
  
 # For each weight configuration, we intialize the MOQ agent and train it for 800 iterations
 #This means if there is 64 weight configurations, we will have 64 agents trained for 800 iterations
  
  for i in range(0, len(weight_combinations)):
    print(i)
    env.reset(seed=seed)
    eval_env.reset(seed=seed)
    scalarization = tchebicheff(tau=6.0, reward_dim=3)
    weights =np.array(weight_combinations[i])
  
    #Instantiate the MOQ alg with the specific weight configuration and Chebyshev Scalarization and Hyperparameters
    agent = MOQLearning(env, scalarization=scalarization,initial_epsilon=1,final_epsilon=0.1,epsilon_decay_steps=800000, gamma=0.99, weights=weights, log=False)

   #Train the agent for 800 iterations
    for z in range(0, 800):
        agent.train(
            total_timesteps=1000,
            reset_num_timesteps= False,
            start_time=time.time(),
            eval_env=eval_env,
        )
       #Test the learnt policy by obtaining the discounted reward at end of each iteration
        _,_,_,disc_reward=(eval_mo(agent, env=eval_env, w=weights))
        moq_eval_rewards[i][z]=disc_reward
        
  #Analyse the Convergence to Pareto front approximation, Hypervolume, Cardinality and Sparsity
  #function defined in Utilities.py
  pf,hypervolume_scores,cardinality_scores,sparsity_scores=eval_unknown(moq_eval_rewards,np.array([-1,-1,-1]),eval_env,0.99)
  
  #Log the results to wandb
  log_unknown_results(pf,hypervolume_scores,cardinality_scores,sparsity_scores,"Research Project Logs V7",exp_name,"Chebyshev Four Room 0.99")
  print("Balanced MOQ Chebyshev Results for seed: ",seed)
  print(pf)
 
  
