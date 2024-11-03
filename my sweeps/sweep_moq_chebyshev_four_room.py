from os import wait3
# Log in to W&B
import wandb
wandb.login()
from morl_baselines.common.pareto import filter_pareto_dominated
from morl_baselines.common.performance_indicators import hypervolume
import matplotlib.pyplot as plt
from morl_baselines.common.performance_indicators import cardinality
from morl_baselines.common.performance_indicators import igd
from morl_baselines.common.performance_indicators import sparsity
from morl_baselines.common.scalarization import weighted_sum


# Define the sweep configuration
sweep_configuration = {
    'method': 'random',  # Chose Random Search, TOO many values to run all possible combinations
    'metric': {'name': 'Hypervolume', 'goal': 'maximize'},  # We seek to optimize Hypervolume
    'parameters': {

          'tau': {'values':[2,4,6]},  #This is the tau value for the Chebyshev Scalarization
          'gamma': {'values':[0.99,0.9]},    
          'w1':{'values': [0.3,0.1,0.7]},
          'w2':{'values': [0.3,0.1]}

    }
}
import time
import numpy as np

def train():
    wandb.init()
    config = wandb.config

    # Initialization
    env = MORecordEpisodeStatistics(mo_gym.make("four-room-v0"), gamma=config.gamma)
    eval_env = MORecordEpisodeStatistics(mo_gym.make("four-room-v0"), gamma=config.gamma)
    scalarization = tchebicheff(config.tau, reward_dim=3)
    w3=1-config.w1-config.w2
    w1=config.w1
    w2=config.w2
    weights = np.array([w1,w2,w3])

    agent = MOQLearning(
        env,
        scalarization=scalarization,
        weights=weights,
        log=False,
        gamma=config.gamma,
        initial_epsilon=1,
        final_epsilon=0.1,
        epsilon_decay_steps=500000,

    )

    # Training
    agent.train(
        total_timesteps=500000,
        start_time=time.time(),
        eval_freq=10000,
        eval_env=eval_env,
    )
    
    #Test Learnt policy By obtaining the discounted reward
    __,_,_,disc_reward=(eval_mo(agent, env=eval_env, w=weights))
    print(f"disc_reward: {disc_reward} shape: {np.shape(disc_reward)}")
    pf=[]
    pf.append(disc_reward)
    pf=list(filter_pareto_dominated(pf))
    reference_point = np.array([-1, -1, -1])  # Ref Point to Calc Hypervolume
    hv=0

    if(len(pf)==0):
        hv=0
    else:
        hv = hypervolume(reference_point,pf)

    wandb.log({'Hypervolume': hv})


# Initialize the sweep and run the algorithm
sweep_id = wandb.sweep(sweep_configuration, project="Research Project V3.0")
wandb.agent(sweep_id, function=train, count=25) # Run 25 trials
wandb.finish()
