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


SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51,52]  # 10 seeds
env = MORecordEpisodeStatistics(mo_gym.make("deep-sea-treasure-concave-v0"), gamma=0.9)
eval_env = MORecordEpisodeStatistics(mo_gym.make("deep-sea-treasure-concave-v0"), gamma=0.9)
for seed in SEEDS:
  print(f"Running experiment with seed {seed}")
  exp_name = f"FINAL GREEDY Linear Experiment G 0.9 with seed {seed}"
  rows, cols = 11, 4000   #11 Agents
  random.seed(seed)
  np.random.seed(seed)
  env.reset(seed=seed)
  eval_env.reset(seed=seed)

  moq_eval_rewards=np.zeros((rows,cols,2))
  for i in range(0, 11):
      print(i)
      
      #scalarization = tchebicheff(tau=4.0, reward_dim=2)
      weights = np.array([1 - (i / 10), i / 10])

      chebyshev = MOQLearning(env, scalarization=weighted_sum,initial_epsilon=0.1,final_epsilon=0.1, gamma=0.9, weights=weights,seed=seed, log=False)

      for z in range(0, 4000):
          chebyshev.train(
              total_timesteps=100,
              reset_num_timesteps= False,
              start_time=time.time(),
              eval_env=eval_env,
          )
          #eval_env.reset()
          _,_,_,disc_reward=(eval_mo(chebyshev, env=eval_env, w=weights))
          moq_eval_rewards[i][z]=disc_reward
  eval_env.reset(seed=seed)
  pf,hypervolume_scores,cardinality_scores,igd_scores,sparsity_scores=evaluate(moq_eval_rewards,np.array([0,-50]),eval_env)
  log_results(pf,hypervolume_scores,cardinality_scores,igd_scores,sparsity_scores,"Research Project Logs V4",exp_name,"MOQ Linear DST with gamma of 0.9")
  
  print("Balanced MOQ Linear Results for seed: ",seed)
  print(pf)

  
