from os import wait3
import wandb
import time
import numpy as np
import time
import gymnasium as gym
import mo_gymnasium as mo_gym
import numpy as np
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

wandb.login(key='e59e68a6a5472c9bbadb6ab9ce6ca89e9fce402e')

# Define the sweep configuration
sweep_configuration = {
    'method': 'grid',  # Choose the sweep method: random, grid, bayes
    'metric': {'name': 'Hypervolume', 'goal': 'maximize'}, 
    'parameters': {
 
          'tau': {'values':[1,2,3,4,5,6,8]},
          'w1':{'values': [0.3,0.5,0.7,0.8,0.9]}, # I want to find max treasure rewards so lets focus on higher values for w1 
          'gamma':{'values':[0.9,0.99]},
    }
}



#We will analyse the quaility of solutions found by algorithms

def train():
    wandb.init()
    config = wandb.config

    # Initialization
    env = MORecordEpisodeStatistics(mo_gym.make("deep-sea-treasure-concave-v0"), gamma=config.gamma )
    eval_env = mo_gym.make("deep-sea-treasure-concave-v0")
    scalarization = tchebicheff(config.tau, reward_dim=2)

    w1=config.w1
    w2=1-w1
    weights = np.array([w1,w2])

    agent = MOQLearning(
        env,
        scalarization=scalarization,
        weights=weights,
        log=True,
        project_name= "MORL Research",
        experiment_name="Sweeps MOQ in DST Concave",
        gamma=config.gamma,
        initial_epsilon=0.1,
        final_epsilon=0.1,

    )

    # Training
    agent.train(
        total_timesteps=200000,
        start_time=time.time(),
        eval_freq=100,
        eval_env=eval_env,
    )
    __,_,_,disc_reward=(eval_mo(agent, env=eval_env, w=weights))
    print(f"disc_reward: {disc_reward} shape: {np.shape(disc_reward)}")
    
    pf=[]
    pf.append(disc_reward)
    pf=list(filter_pareto_dominated(pf))
    reference_point = np.array([0,-50]) 
    hv=0

    if(len(pf)==0):
        hv=0
    else:
      hv = hypervolume(reference_point,pf)

    wandb.log({'Hypervolume': hv})
    message = f"Reward obtained by Algorithm with gamma={config.gamma}, tau={config.tau}, and weights w1={w1}, w2={w2} is {disc_reward}. with hypervolume of {hv}"
    print(message)

  # Open a file in append mode ('a' allows you to keep adding to the file without overwriting)
    with open("results.txt", "a") as file:
      file.write(message + "\n")  # Write the message and add a newline character


# Initialize the sweep and run the agent
sweep_id = wandb.sweep(sweep_configuration, project="MORL Research")
wandb.agent(sweep_id, function=train, count=80)  
wandb.finish()
