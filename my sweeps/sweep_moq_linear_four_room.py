from os import wait3
# Log in to W&B
import wandb
wandb.login()

# sweep configuration is defined
sweep_configuration = {
    'method': 'grid',  # Used Grid Search since we had few configurations
    'metric': {'name': 'Hypervolume', 'goal': 'maximize'},  # We seek to maximise the Hypervolume
    'parameters': {

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
    
    w3=1-config.w1-config.w2
    w1=config.w1
    w2=config.w2
    weights = np.array([w1,w2,w3])

    agent = MOQLearning(
        env,
        scalarization=weighted_sum,
        weights=weights,
        log=True,
        gamma=config.gamma,
        initial_epsilon=1,
        final_epsilon=0.1,
        epsilon_decay_steps=500000,

    )

    # Training
    agent.train(
        total_timesteps=500000,
        start_time=time.time(),
        eval_freq=1000,
        eval_env=eval_env,
    )
    
    
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


# Initialize the sweep and run the agent
sweep_id = wandb.sweep(sweep_configuration, project="Research Project V3.0")
wandb.agent(sweep_id, function=train, count=12)  #12 Runs
wandb.finish()
