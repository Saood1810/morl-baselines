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


SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]  # 10 seeds

#Environment Instantiation
env = MORecordEpisodeStatistics(mo_gym.make("four-room-v0"), gamma=0.99)
eval_env = MORecordEpisodeStatistics(mo_gym.make("four-room-v0"), gamma=0.99)

# Run the experiment for multiple seeds
for seed in SEEDS:
  # Generate weight combinations  so that we can run the MO Q as an outer-loop method
  weight_combinations = generate_combinations()
  
  print(f"Running experiment with seed {seed}")
  exp_name = f"MOQ Linear Four Room {seed}"
  

  rows, cols = len(weight_combinations), 800 # No of configurations and iterations
  random.seed(seed)
  np.random.seed(seed)
  env.reset(seed=seed)
  eval_env.reset(seed=seed)

  # Numpy Array to store the results of each weight configuration across 800 iterations
  moq_eval_rewards=np.zeros((rows,cols,3)) 

    #For a single experiment, we run the MO Q learning algorithm for each weight configuration to get combined assessment of algorithm..
    # this makes it an outer-loop method
  for i in range(0, len(weight_combinations)):
    print(i)
    env.reset(seed=seed)
    eval_env.reset(seed=seed)
        
    weights =np.array(weight_combinations[i])

    #Algorithm Instantiation with Hyperparameters and weights
    agent = MOQLearning(env, scalarization=weighted_sum,initial_epsilon=1,final_epsilon=0.1,epsilon_decay_steps=800000, gamma=0.99, weights=weights, log=False)

    # We train algorithms every 1000 timesteps for 800 iterations... at end of each iteration, we evaluate the algorithm
    for z in range(0, 800):
        agent.train(
            total_timesteps=1000,
            reset_num_timesteps= False,
            start_time=time.time(),
            eval_env=eval_env,
        )
        
       # We evaluate the algorithm by obtaining the discounted reward
        _,_,_,disc_reward=(eval_mo(agent, env=eval_env, w=weights))
        moq_eval_rewards[i][z]=disc_reward #Store the reward of 
        
  #Now that we got the rewards at each iteration for each configuration
  # We Evaluate the performance of the MOQ algorithms across all weight configurations by calculating the pf approximation .Hypervolume, Cardinality, IGD and Sparsity
  pf,hypervolume_scores,cardinality_scores,sparsity_scores=eval_unknown(moq_eval_rewards,np.array([-1,-1,-1]),eval_env,0.99)
  
  #Plot results to wandb
  log_unknown_results(pf,hypervolume_scores,cardinality_scores,sparsity_scores,"Research Project Logs V7",exp_name,"Linear Four Room 0.99")
  print("Balanced MOQ Linear Results for seed: ",seed)
  print(pf)
 
  
