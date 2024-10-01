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

def generate_combinations(step=0.1):
    weights = [round(i * step, 1) for i in range(int(1 / step) + 1)]
    combinations = []
    for w1 in weights:
        for w2 in weights:
            w3 = 1 - w1 - w2
            if 0 <= w3 <= 1:
                combinations.append((w1, w2, w3))
    return combinations



# will have exp_name as a parameter
from pymoo.visualization.scatter import Scatter
from pymoo.problems import get_problem

def eval_pql(tracked_policies,ref_point,eval_env,gamma):
  hypervolume_scores = [0]
  igd_scores = [0]
  sparsity_scores = [0]
  cardinality_scores = [0]
  true_pf = eval_env.pareto_front(gamma=gamma)
  
  
  for policy in tracked_policies:
    pf = policy
    pf = list(filter_pareto_dominated(pf))
    if len(pf) > 0:
        hypervolume_scores.append(hypervolume(ref_point, pf))
        cardinality_scores.append(cardinality(pf))
        igd_scores.append(igd(true_pf, pf))
        sparsity_scores.append(sparsity(pf))
        
  return pf,hypervolume_scores,cardinality_scores,igd_scores,sparsity_scores
  



def evaluate(tracked_policies,ref_point,eval_env,gamma):


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

      # Filter Pareto front and calculate hypervolume
      pf = list(filter_pareto_dominated(pf))
      if len(pf) > 0:
          hypervolume_scores.append(hypervolume(ref_point, pf))
          cardinality_scores.append(cardinality(pf))
          igd_scores.append(igd(true_pf, pf))
          sparsity_scores.append(sparsity(pf))

  return pf,hypervolume_scores,cardinality_scores,igd_scores,sparsity_scores


def eval_unknown(tracked_policies,ref_point,eval_env,gamma):
  
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
  
  
  
def log_results(pf, hypervolume_scores,cardinality_scores,igd_scores,sparsity_scores,proj_name,exp_name,group,total_timesteps):

  wandb.init(mode="offline",project=proj_name,group=group,name=exp_name)
  timesteps=[0]
  for i in range(len(hypervolume_scores)):
    timesteps.append((i+1)) #Tracking every 100 steps
  
  '''for i in range(0,total_timesteps):
    wandb.log({"global_step": i})'''

  # Log each score set to wandb
  #wandb.log({"Hypervolume": hypervolume_scores, "Cardinality":cardinality_scores,"IGD":igd_scores,"Sparsity":sparsity_scores})
   # Log each score set to wandb
  for i, (timestep,hv_score, cd_score, igd_score, sp_score) in enumerate(zip(timesteps,hypervolume_scores, cardinality_scores, igd_scores, sparsity_scores)):
        wandb.log({

            'Hypervolume': hv_score,
            'Cardinality': cd_score,
            'IGD': igd_score,
            'Sparsity': sp_score,
            'Timesteps': timestep,
            'global_step': timestep*100,
        })
  wandb.finish()


  # Print results
  print("Final Pareto Front:", pf)
  '''print("Hypervolume Scores:", hypervolume_scores)
  # Print results


  # Plotting the hypervolume scores over time
  plt.plot(hypervolume_scores)
  plt.xlabel('Timesteps')
  plt.ylabel('Hypervolume')
  plt.title('Learning Curve with Hypervolume')
  plt.grid(True)

  #wandb.log({"roc": wandb.plot.roc_curve(hypervolume_scores)})
  plt.show()


  plt.plot(cardinality_scores)
  plt.xlabel('Timesteps')
  plt.ylabel('Cardinality')
  plt.title('Learning Curve with Cardinality')
  plt.grid(True)

  #wandb.log({"roc": wandb.plot.roc_curve(cardinality_scores)})
  plt.show()

  plt.plot(sparsity_scores)
  plt.xlabel('Timesteps')
  plt.ylabel('Sparisty')
  plt.title('Learning Curve with Sparsity')
  plt.grid(True)

  plt.show()

  plt.plot(igd_scores)
  plt.xlabel('Timesteps')
  plt.ylabel('IGD')
  plt.title('Learning Curve with IGD')
  plt.grid(True)

  plt.show()'''




  
def scatter_plot_known_pareto(pf1,pf2,eval_env,ref_point,exp_name,alg_name1,alg_name2):
  

  # Pareto front and hypervolume calculations
  true_pf = eval_env.pareto_front(gamma=0.99)
 
  # Scatter plot using pymoo
  plot = Scatter().add(np.array(true_pf),color="blue", marker="o", label="True Pareto Front")
  plot.add(np.array(pf1), color="green",marker="X" ,label=alg_name1)

  plot.add(np.array(pf2), color="red",marker="*", label=alg_name2)
  plot.legend=True

  # Save the plot to a file
  plot.show()
  file_name=exp_name+".png"
  plt.savefig(file_name)

  # Log the plot image to wandb
  wandb.log({"Pareto Front": wandb.Image(file_name)})





