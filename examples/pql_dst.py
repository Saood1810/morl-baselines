import mo_gymnasium as mo_gym
import numpy as np

from morl_baselines.multi_policy.pareto_q_learning.pql import PQL
from utilities import eval_pql,log_results

if __name__ == "__main__":
    env = mo_gym.make("deep-sea-treasure-concave-v0")
    ref_point = np.array([0, -25])

    agent = PQL(
        env,
        ref_point,
        gamma=0.99,
        initial_epsilon=1.0,
        epsilon_decay_steps=50000,
        final_epsilon=0.2,
        seed=1,
        project_name="MORL-Baselines",
        experiment_name="Pareto Q-Learning",
        log=True,
    )

    pf,tracked_policies = agent.train(
        total_timesteps=10000,
        log_every=100,
        action_eval="hypervolume",
        known_pareto_front=env.pareto_front(gamma=0.99),
        ref_point=ref_point,
        eval_env=env,
    )
    print(pf)
    
    pf_approx,hypervolume_scores,cardinality_scores,igd_scores,sparsity_scores=eval_pql(tracked_policies,ref_point,env,agent.gamma)
    log_results(pf_approx,hypervolume_scores,cardinality_scores,igd_scores,sparsity_scores,"MORL-Baselines","Pareto Q-Learning","Pareto Q-Learning")
    

    # Execute a policy
    target = np.array(pf.pop())
    print(f"Tracking {target}")
    reward = agent.track_policy(target, env=env)
    print(f"Obtained {reward}")
