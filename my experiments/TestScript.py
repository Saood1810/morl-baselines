# -*- coding: utf-8 -*-
"""TestScript.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/133PShChL88LA0Xeto4wkR4gNxsFC7xN5
"""

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
from pymoo.indicators.hv import HV
from pymoo.visualization.scatter import Scatter

import numpy as np
import time
import matplotlib.pyplot as plt
from morl_baselines.common.evaluation import policy_evaluation_mo
wandb.login(key='e59e68a6a5472c9bbadb6ab9ce6ca89e9fce402e')
wandb.init(project="CHPC Results")
def generate_combinations(step=0.1):
    weights = [round(i * step, 1) for i in range(int(1 / step) + 1)]
    combinations = []
    for w1 in weights:
        for w2 in weights:
            w3 = 1 - w1 - w2
            if 0 <= w3 <= 1:
                combinations.append((w1, w2, w3))
    return combinations

# Generate and print combinations
combinations = generate_combinations()
print(len(combinations))
for combo in combinations:
    print(combo)

weight_combinations = generate_combinations()
rows, cols = len(weight_combinations), 2000   #64 Agents and
moq_eval_rewards=np.zeros((rows,cols,3))

for i in range(0, len(weight_combinations)):
    print(i)
    env = MORecordEpisodeStatistics(mo_gym.make("resource-gathering-v0"), gamma=0.99)
    eval_env = mo_gym.make("resource-gathering-v0", render_mode="rgb_array")
        #scalarization = tchebicheff(tau=4.0, reward_dim=2)
    weights =np.array(weight_combinations[i])

    agent = MOQLearning(env, scalarization=weighted_sum,initial_epsilon=0.1,final_epsilon=0.1, gamma=0.99, weights=weights, log=True)

    for z in range(0, 2000):
        agent.train(
            total_timesteps=100,
            reset_num_timesteps= False,
            start_time=time.time(),
            eval_freq=100,
            eval_env=eval_env,
        )
        #eval_env.reset()
        _,_,_,disc_reward=(eval_mo(agent, env=eval_env, w=weights))
        moq_eval_rewards[i][z]=disc_reward


eval_env = mo_gym.make("resource-gathering-v0", render_mode="rgb_array")
pf,hypervolume_scores,cardinality_scores,igd_scores,sparsity_scores=eval(moq_eval_rewards,np.array([-1,-1,-2]),eval_env)
print(pf)
print(hypervolume_scores)
print(cardinality_scores)
print(igd_scores)
print(sparsity_scores)
#log_results(pf,hypervolume_scores,cardinality_scores,igd_scores,sparsity_scores,"Greedy MO Q Learning with Chebyshev Scalarization Tau 5")