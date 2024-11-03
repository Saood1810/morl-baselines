from os import wait3
# Log in to W&B
import wandb
wandb.login()

# sweep configuration is Defined
sweep_configuration = {
    'method': 'random',  # Chose random search
    'metric': {'name': 'Hypervolume', 'goal': 'maximize'},  # We seek to maximize the Hypervolume
    'parameters': {
          'tau': {'values':[2,4,6]},
          
          'total_timesteps':{'values': [100000,200000,300000]},
          'gamma': {'values':[0.99,0.9]},
          'w1':{'values': [0.9,0.5,0.1]}

    }
}

# Define the training function
import time
import numpy as np


def train():
    wandb.init()
    config = wandb.config

    # Initialization
    env = MORecordEpisodeStatistics(mo_gym.make("deep-sea-treasure-concave-v0"), gamma=config.gamma)
    eval_env = MORecordEpisodeStatistics(mo_gym.make("deep-sea-treasure-concave-v0"), gamma=config.gamma)
    w1=config.w1
    w2=1-config.w1
    weights = np.array([w1,w2])
   

    agent = MOQLearning(
        env,
        scalarization=weighted_sum,
        weights=weights,
        log=False,
        gamma=config.gamma,
        initial_epsilon=1,
        final_epsilon=0.1,
        epsilon_decay_steps=config.total_timesteps,

    )

    # Training
    agent.train(
        total_timesteps=config.total_timesteps,
        start_time=time.time(),
        eval_freq=100000,
        eval_env=eval_env,
    )
    
    #Test learnt policy by obtaining the discounted reward
    __,_,_,disc_reward=(eval_mo(agent, env=eval_env, w=weights))
    print(f"disc_reward: {disc_reward} shape: {np.shape(disc_reward)}")
    pf=[]
    pf.append(disc_reward)
    pf=list(filter_pareto_dominated(pf)) # Will filter solution out if not optimal
    reference_point = np.array([0,-50])  # Reference Point to Calculate Hypervolume
    hv=0

    if(len(pf)==0):
        hv=0
    else:
        hv = hypervolume(reference_point,pf) #Calculate Hypervolume 

    wandb.log({'Hypervolume': hv}) #Log score to wandb


# Initialize the sweep and run the agent
sweep_id = wandb.sweep(sweep_configuration, project="Research Project V4.0")
wandb.agent(sweep_id, function=train, count=25)  # 25 Runs
wandb.finish()
