import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import gym_electric_motor as gem
from gym_electric_motor.constraint_monitor import ConstraintMonitor
from gym_electric_motor.reference_generators import \
    MultipleReferenceGenerator,\
    WienerProcessReferenceGenerator
from gym_electric_motor.visualization import MotorDashboard
from gym_electric_motor.physical_systems import ConstantSpeedLoad

from gym.spaces import Discrete, Box
from gym.wrappers import FlattenObservation, TimeLimit
from gym import ObservationWrapper

from tensorforce.environments import Environment
from tensorforce.agents import Agent
import json


def save_to_json(data, name, indent=4, sort=False, ending='.json'):
    """"""

    with open(name + ending, 'w') as json_file:
        json.dump(data, json_file, indent=indent, sort_keys=sort)
    print('Done writing')


def load_json(filepath):
    """"""
    data = json.loads(filepath)
    print('Loaded')
    return data


def set_plot_params(figure_title=22, figsize=(20, 14), title_size=24,
                    label_size=20, tick_size=16, marker_size=10, line_width=3,
                    legend_font=16, style='dark'):
    """ setting matplotlib params """

    if style == 'dark':
        plt.style.use('dark_background')
    else:
        plt.style.use(style)

    params = {'axes.titlesize': title_size,
              'legend.fontsize': legend_font,
              'figure.figsize': figsize,
              'axes.labelsize': label_size,
              'xtick.labelsize': label_size,
              'ytick.labelsize': label_size,
              'figure.titlesize': figure_title}
    plt.rcParams.update(params)


class SqdCurrentMonitor:
    """
    monitor for squared currents:

    i_sd**2 + i_sq**2 < 1.5 * nominal_limit
    """

    def __call__(self, state, observed_states, k, physical_system):
        self.I_SD_IDX = physical_system.state_names.index('i_sd')
        self.I_SQ_IDX = physical_system.state_names.index('i_sq')
        # normalize to limit_values, since state is normalized
        nominal_values = physical_system.nominal_state / abs(
            physical_system.limits)
        limits = 1.5 * nominal_values
        # calculating squared currents as observed measure
        sqd_currents = state[self.I_SD_IDX] ** 2 + state[self.I_SQ_IDX] ** 2

        return (sqd_currents > limits[self.I_SD_IDX] or sqd_currents > limits[
            self.I_SQ_IDX])


class EpsilonWrapper(ObservationWrapper):
    """
    Changes Epsilon in a flattened observation to cos(epsilon)
    and sin(epsilon)
    """

    def __init__(self, env, epsilon_idx):
        super(EpsilonWrapper, self).__init__(env)
        self.EPSILON_IDX = epsilon_idx
        new_low = np.concatenate((self.env.observation_space.low[
                                  :self.EPSILON_IDX], np.array([-1.]),
                                  self.env.observation_space.low[
                                  self.EPSILON_IDX:]))
        new_high = np.concatenate((self.env.observation_space.high[
                                   :self.EPSILON_IDX], np.array([1.]),
                                   self.env.observation_space.high[
                                   self.EPSILON_IDX:]))

        self.observation_space = Box(new_low, new_high)

    def observation(self, observation):
        cos_eps = np.cos(observation[self.EPSILON_IDX] * np.pi)
        sin_eps = np.sin(observation[self.EPSILON_IDX] * np.pi)
        observation = np.concatenate((observation[:self.EPSILON_IDX],
                                      np.array([cos_eps, sin_eps]),
                                      observation[self.EPSILON_IDX + 1:]))
        return observation


# config_path = '/home/pascal/Sciebo/Uni/Master/Semester_2/Projektarbeit/' + \
#               'python/saves/configs'
# agent_path = '/home/pascal/Sciebo/Uni/Master/Semester_2/Projektarbeit/' + \
#              'python/saves/agents'

config_path = '/home/student/hdd1/ppeters/saved_configs/'
agent_path = '/home/student/hdd1/ppeters/saved_agents/'

print('env name: ')
env_name = input() #'env_config_zero_init'
agent_config_name = 'default_agent'

print('agent name: ')
agent_name = input() #'dqn_default'

# define motor arguments
load_initializer = {'interval': [[-4000*2*np.pi/60, 4000*2*np.pi/60]],
                    'random_init': 'uniform'}

sqd_current_monitor = ConstraintMonitor(external_monitor=SqdCurrentMonitor)
const_random_load = ConstantSpeedLoad(load_initializer=load_initializer)
q_generator = WienerProcessReferenceGenerator(reference_state='i_sq')
d_generator = WienerProcessReferenceGenerator(reference_state='i_sd')
rg = MultipleReferenceGenerator([q_generator, d_generator])
max_eps_steps = 10000
simulation_steps = 500000

environment_config = load_json(config_path + env_name)
# creating gem environment
gem_env = gem.make(
        "PMSMDisc-v1",
        visualization=MotorDashboard(plots=['i_sq', 'i_sd', 'reward']),
        reward_function=gem.reward_functions.WeightedSumOfErrors(
                    observed_states=['i_sq', 'i_sd'],
                    reward_weights={'i_sq': 1, 'i_sd': 1},
                    constraint_monitor=SqdCurrentMonitor(),
                    gamma=0.99,
                    reward_power=1),
        reference_generator=rg,
        **environment_config)

# appling wrappers and modifying environment
gem_env.action_space = Discrete(7)
eps_idx = gem_env.physical_system.state_names.index('epsilon')
#gem_env_ = TimeLimit(EpsilonWrapper(FlattenObservation(gem_env), eps_idx),
#                     max_eps_steps)
gem_env_ = EpsilonWrapper(FlattenObservation(gem_env), eps_idx)
# creating tensorforce environment
tensor_env = Environment.create(environment=gem_env_,
                                max_episode_timesteps=max_eps_steps)


agent_config = load_json(config_path + agent_config_name)

dqn_agent = Agent.load(directory=agent_path,
                       filename=agent_name,
                       environment=tensor_env,
                       **agent_config)
                       # agent='dqn',
                       # memory=200000,
                       # batch_size=25,
                       # network=net,
                       # update_frequency=1,
                       # start_updating=10000,
                       # learning_rate=1e-4,
                       # discount=0.99,
                       # exploration=epsilon_decay,
                       # target_sync_frequency=1000,
                       # target_update_weight=1.0,)

print('\n ... agent loaded ...\n')

# test agent
tau = 1e-5
steps = 1000000

rewards = []
lens = []
obs_hist = []

states = []
references = []

obs = gem_env_.reset()
obs_hist.append(obs)
terminal = False
cum_rew = 0
step_counter = 0
eps_rew = 0

for step in tqdm(range(steps)):
    #while not terminal:
    gem_env_.render()
    actions = dqn_agent.act(states=obs, independent=True)

    obs, reward, terminal, _ = gem_env_.step(action=actions)
    rewards.append(cum_rew)
    obs_hist.append(obs)
    # dqn_agent.observe(terminal, reward=reward)
    cum_rew += reward
    eps_rew += reward

    if terminal:
        lens.append(step_counter)
        step_counter = 0
        #print(f'Episode length: {episode_length} steps')
        obs = gem_env_.reset()

        obs_hist.append(obs)
        rewards.append(eps_rew)

        terminal = False
        eps_rew = 0
    step_counter += 1


print(f' \n Cumulated Reward over {steps} steps is {cum_rew/steps} \n')
#print(f' \n Longest Episode: {np.amax(lens)} steps \n')
#obs_hist = np.asarray(obs_hist)


#idx_isd = gem_env.physical_system.state_positions['i_sd']
#idx_isq = gem_env.physical_system.state_positions['i_sq']

#print('num obs', len(obs_hist))

# set_plot_params()
# fig1, axs1 = plt.subplots(2, 1, sharex=True)
# axs1[0].set_title('reward and lens')
# axs1[0].plot(range(len(rewards)), rewards)
# axs1[1].plot(range(len(lens)), lens)
#
# fig2, axs2 = plt.subplots(2, 1, sharex=True)
# axs2[0].set_title('obs')
# axs2[0].plot(range(len(obs_hist[:, idx_isd])), obs_hist[:, idx_isd])
# axs2[1].plot(range(len(obs_hist[:, idx_isq])), obs_hist[:, idx_isq])
# axs2[0].hlines(episode_length, 0, 1, 'r')
# axs2[0].hlines(episode_length, 0, 1, 'r')
#
# plt.show()
