"""Script for performing learning: Resume environment, initialise agent (and load model if applicable),
 initialise runner, run and print statistics """

from printind.printind_function import printi, printiv
from env import resume_env, nb_actuations

import os
import numpy as np
from tensorforce.agents import PPOAgent
from tensorforce.execution import Runner

"""
import sys
import os
cwd = os.getcwd()
sys.path.append(cwd + "/../Simulation/")

from Env2DCylinder import Env2DCylinder
"""

printi("resume env")

# Resume environment
environment = resume_env(plot=200, step=100, dump=100)

printi("define network specs")

# Define NN specs to be passed to the agent
network_spec = [
    dict(type='dense', size=512),
    dict(type='dense', size=512),
]

printi("define agent")

printiv(environment.states)
printiv(environment.actions)
printiv(network_spec)

# Initialise PPO agent
agent = PPOAgent(
    states=environment.states, # The state-space description dictionary.
    actions=environment.actions, # The action-space description dictionary.
    network=network_spec,
    # Agent
    states_preprocessing=None, # Dict specifying whether and how to preprocess state signals (e.g. normalization, greyscale, etc..)
    actions_exploration=None, # Dict specifying whether and how to add exploration to the model's "action outputs" (e.g. epsilon-greedy).
    reward_preprocessing=None, # Dict specifying whether and how to preprocess rewards coming from the Environment (e.g. reward normalization).
    # MemoryModel
    update_mode=dict(
        unit='episodes',
        # 10 episodes per update
        batch_size=20,
        # Every 10 episodes
        frequency=20
    ),
    memory=dict(
        type='latest',
        include_next_states=False,
        capacity=10000
    ),
    # DistributionModel
    distributions=None,
    entropy_regularization=0.01,
    # PGModel
    baseline_mode='states',
    baseline=dict(
        type='mlp',
        sizes=[32, 32]
    ),
    baseline_optimizer=dict(
        type='multi_step',
        optimizer=dict(
            type='adam',
            learning_rate=1e-3
        ),
        num_steps=5
    ),
    gae_lambda=0.97,
    # PGLRModel
    likelihood_ratio_clipping=0.2,
    # PPOAgent
    step_optimizer=dict(
        type='adam',
        learning_rate=1e-3
    ),
    subsampling_fraction=0.2,
    optimization_steps=25,
    execution=dict(
        type='single',
        session_config=None,
        distributed_spec=None
    )
)

# Restore agent's TensorFlow model (NN setup), if a checkpoint exists. Otherwise do nothing.
# Checkpoint will only exist if runs have been made
if(os.path.exists('saved_models/checkpoint')): # "checkpoint" file simply keeps a record of latest checkpoint files saved
    restore_path = './saved_models'
else:
    restore_path = None

# If there is a checkpoint, we use it to restore the model
if restore_path is not None:
    printi("restore the model")
    agent.restore_model(restore_path) # If no checkpoint file is given, the latest checkpoint is restored.


# Define the runner
printi("define runner")
runner = Runner(agent=agent, environment=environment)

# Callback function printing episode statistics and saving agent's Tf model after each episode
def episode_finished(r):
    print("Finished episode {ep} after {ts} timesteps (Last reward: {reward})".format(ep=r.episode, ts=r.episode_timestep,
                                                                                 reward=r.episode_rewards[-1]))
    # Save TensorFlow model (NN setup)
    printi("save the mode")

    # If append_timestep = True : Appends current timestep to prevent overwriting previous checkpoint files.
    # If append_timestep = False : To be able to load model from the same given path argument (./saved_models) as given
    #  here (without checkpoint timestep suffix)

    name_save = "./saved_models/ppo_model" # Checkpoint files will be saved under saved models, with name ppo_model.*
    # NOTE: need to check if should create the dir
    r.agent.save_model(name_save, append_timestep=False)

    # Uncomment for plotting
    # r.environment.show_control()
    # r.environment.show_drag()

    # print(sess.run(tf.global_variables()))

    return True


# Start learning
printi("start learning")

runner.run(episodes=200, max_episode_timesteps=nb_actuations, episode_finished=episode_finished)
# For testing (too few timesteps) :
# runner.run(episodes=10000, max_episode_timesteps=20, episode_finished=episode_finished)

runner.close()

# Print learning statistics
print("Learning finished. Total episodes: {ep}. Average reward of last 100 episodes: {ar}.".format(
    ep=runner.episode,
    ar=np.mean(runner.episode_rewards[-100:]))
)
