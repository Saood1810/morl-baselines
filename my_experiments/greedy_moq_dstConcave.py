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
from pymoo.indicators.hv import HV
from pymoo.visualization.scatter import Scatter

import numpy as np
import time
import matplotlib.pyplot as plt
from morl_baselines.common.evaluation import policy_evaluation_mo


rows, cols = 11, 2   #11 Agents and eval at 5 steps each
moq_eval_rewards=np.zeros((rows,cols,2))
for i in range(0, 11):
    print(i)
    env = MORecordEpisodeStatistics(mo_gym.make("deep-sea-treasure-concave-v0"), gamma=0.99)
    eval_env = mo_gym.make("deep-sea-treasure-concave-v0", render_mode="rgb_array")
    #scalarization = tchebicheff(tau=4.0, reward_dim=2)
    weights = np.array([1 - (i / 10), i / 10])

    moq = MOQLearning(env, scalarization=tchebicheff(tau=4.0, reward_dim=2),initial_epsilon=0.1,final_epsilon=0.1, gamma=0.99, weights=weights, log=False)

    for z in range(0, 2):
        moq.train(
            total_timesteps=100,
            reset_num_timesteps= False,
            start_time=time.time(),
            eval_freq=100,
            eval_env=eval_env,
        )
        #eval_env.reset()
        _,_,_,disc_reward=(eval_mo(moq, env=eval_env, w=weights))
        moq_eval_rewards[i][z]=disc_reward
eval_env = mo_gym.make("deep-sea-treasure-concave-v0", render_mode="rgb_array")
pf,hypervolume_scores,cardinality_scores,igd_scores,sparsity_scores=evaluate(moq_eval_rewards,np.array([0,-50]),eval_env)
log_results(pf,hypervolume_scores,cardinality_scores,igd_scores,sparsity_scores,"Greedy MO Q Learning with Greedy test Scalarization")