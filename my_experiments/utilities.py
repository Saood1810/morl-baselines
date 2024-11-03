from morl_baselines.common.pareto import filter_pareto_dominated
from morl_baselines.common.performance_indicators import hypervolume
import matplotlib.pyplot as plt
from morl_baselines.common.performance_indicators import cardinality
from morl_baselines.common.performance_indicators import igd
from morl_baselines.common.performance_indicators import sparsity
from morl_baselines.common.scalarization import weighted_sum
from pymoo.indicators.hv import HV
from pymoo.visualization.scatter import Scatter
from pymoo.problems import get_problem

import wandb

#wandb.init(mode="offline", project="CHPC Results")
import numpy as np


#Function to generate weight combinations for 3 objectives
def generate_combinations(step=0.1):
    weights = [round(i * step, 1) for i in range(int(1 / step) + 1)]
    combinations = []
    for w1 in weights:
        for w2 in weights:
            w3 = 1 - w1 - w2
            if 0 <= w3 <= 1:
                combinations.append((w1, w2, w3))
    return combinations



#Function to evaluate the performance of the algorithm
def evaluate(tracked_policies,ref_point,eval_env,gamma):

#Tracked policies is a set which stores the discounted rewards of the policies every 1000 steps

# Essentially from the tracked policies we are calculating the Hypervolume, Cardinality, IGD and Sparsity
#This helps analyse the convergence to the Approximated Pareto fronts
  hypervolume_scores = [0]
  hypervolume_scores=[0]
  igd_scores=[0]
  sparsity_scores=[0]
  cardinality_scores=[0]
  true_pf=eval_env.pareto_front(gamma=gamma)
  # Number of columns (timesteps)
  num_columns = tracked_policies.shape[1]
  print(num_columns)

  # Iterate over each timestep
  for i in range(num_columns):
      # Extract policies at this timestep for all agents
      pf = tracked_policies[:, i, :].tolist()

      # Filter Pareto front and calculate hypervolume,cadinality,igd and sparsity
      pf = list(filter_pareto_dominated(pf))
      if len(pf) > 0:
          hypervolume_scores.append(hypervolume(ref_point, pf))
          cardinality_scores.append(cardinality(pf))
          igd_scores.append(igd(true_pf, pf))
          sparsity_scores.append(sparsity(pf))

  return pf,hypervolume_scores,cardinality_scores,igd_scores,sparsity_scores

# Function to evaluate the performance of the algorithm when the Pareto Front is not known
def eval_unknown(tracked_policies,ref_point,eval_env,gamma):
  
  #Function is same as above except it does not compute the IGD
  hypervolume_scores=[0]
  
  sparsity_scores=[0]
  cardinality_scores=[0]
  # Number of columns (timesteps)
  num_columns = tracked_policies.shape[1]
  print(num_columns)

  # Iterate over each timestep
  for i in range(num_columns):
      # Extract policies at this timestep for all agents
      pf = tracked_policies[:, i, :].tolist()

      # Filter Pareto front and calculate hypervolume
      pf = list(filter_pareto_dominated(pf))
      if len(pf) > 0:
          hypervolume_scores.append(hypervolume(ref_point, pf))
          cardinality_scores.append(cardinality(pf))
          sparsity_scores.append(sparsity(pf))

  return pf,hypervolume_scores,cardinality_scores,sparsity_scores

#Plot the results of the algorithms which have been evaluated in environments with unknown Pareto Fronts
def log_unknown_results(pf, hypervolume_scores,cardinality_scores,sparsity_scores,proj_name,exp_name,group):
  wandb.init(mode="offline",project=proj_name,group=group,name=exp_name)
  timesteps=[0]
  for i in range(len(hypervolume_scores)):
    timesteps.append((i+1)*100) #Tracking every 100 steps

  # Log each score set to wandb
  #wandb.log({"Hypervolume": hypervolume_scores, "Cardinality":cardinality_scores,"IGD":igd_scores,"Sparsity":sparsity_scores})
   # Log each score set to wandb
  for i, (timestep,hv_score, cd_score, sp_score) in enumerate(zip(timesteps,hypervolume_scores, cardinality_scores, sparsity_scores)):
        wandb.log({

            'hypervolume': hv_score,
            'cardinality': cd_score,   
            'sparsity': sp_score,
            'global_step': timestep,
        })
  wandb.finish()


  # Print results
  print("Final Pareto Front:", pf)
  
  
#Plot the results of the algorithms which have been evaluated in environments with known Pareto Fronts
def log_results(pf, hypervolume_scores,cardinality_scores,igd_scores,sparsity_scores,proj_name,exp_name,group):

  wandb.init(mode="offline",project=proj_name,group=group,name=exp_name)
  timesteps=[0]
  for i in range(len(hypervolume_scores)):
    timesteps.append((i+1)) #Tracking every 100 steps
  
  '''for i in range(0,total_timesteps):
    wandb.log({"global_step": i})'''

   # Log each score set to wandb
  for i, (timestep,hv_score, cd_score, igd_score, sp_score) in enumerate(zip(timesteps,hypervolume_scores, cardinality_scores, igd_scores, sparsity_scores)):
        wandb.log({

            'Hypervolume': hv_score,
            'Cardinality': cd_score,
            'IGD': igd_score,
            'Sparsity': sp_score,
            'Timesteps': timestep,
            'global_step': timestep*1000,
        })
  wandb.finish()
  

      
      
  





