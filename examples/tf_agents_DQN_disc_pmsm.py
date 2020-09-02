import gym_electric_motor as gem
import gym
import tensorflow as tf
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.networks import q_network
from gym.wrappers import FlattenObservation  # , TimeLimit
from tf_agents.environments.wrappers import TimeLimit
from tf_agents.agents.dqn import dqn_agent
from tf_agents.utils import common

from gym_electric_motor.visualization import MotorDashboard, ConsolePrinter
from gym_electric_motor.physical_systems import ConstantSpeedLoad
from gym import ObservationWrapper
from gym.spaces import Discrete, Box
import numpy as np
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.utils.common import function
import time

gamma = 0.99
tau = 1e-5


class SqdCurrentMonitor:
    """
    monitor for squared currents:

    i_sd**2 + i_sq**2 < 1.5 * nominal_limit
    """

    def __call__(self, state, observed_states, k, physical_system):
        self.I_SD_IDX = physical_system.state_names.index('i_sd')
        self.I_SQ_IDX = physical_system.state_names.index('i_sq')
        # normalize to limit_values, since state is normalized
        nominal_values = physical_system.nominal_state / abs(physical_system.limits)
        limits = 1.5 * nominal_values
        # calculating squared currents as observed measure
        sqd_currents = state[self.I_SD_IDX] ** 2 + state[self.I_SQ_IDX] ** 2

        return (sqd_currents > limits[self.I_SD_IDX] or sqd_currents > limits[self.I_SQ_IDX])


motor_parameter = dict(p=3,  # [p] = 1, nb of pole pairs
                       r_s=17.932e-3,  # [r_s] = Ohm, stator resistance
                       l_d=0.37e-3,  # [l_d] = H, d-axis inductance
                       l_q=1.2e-3,  # [l_q] = H, q-axis inductance
                       psi_p=65.65e-3,  # [psi_p] = Vs, magnetic flux of the permanent magnet
                       )  # BRUSA
u_sup = 350

nominal_values = dict(omega=4000 * 2 * np.pi / 60,
                      i=230,
                      u=u_sup
                      )
limit_values = {key: 1.3 * nomin for key, nomin in nominal_values.items()}
# limit_values=nominal_values.copy()

q_generator = gem.reference_generators.WienerProcessReferenceGenerator(reference_state='i_sq')
d_generator = gem.reference_generators.WienerProcessReferenceGenerator(reference_state='i_sd')
rg = gem.reference_generators.MultipleReferenceGenerator([q_generator, d_generator])

# Kwargs for training environment
gym_env_kwargs1 = {'visualization': MotorDashboard(plots=['i_sq', 'i_sd', 'reward']),
                   # parameterize the PMSM
                   'motor_parameter': motor_parameter,
                   'limit_values': limit_values,
                   'nominal_values': nominal_values,
                   'u_sup': u_sup,
                   'load': 'ConstSpeedLoad',  # ConstantSpeedLoad(omega_fixed=1000 * np.pi / 30),
                   'load_initializer': {'random_init': 'uniform', },
                   'tau': tau,
                   'motor_initializer': {'random_init': 'uniform', },

                   ## pass a reward function with a gamma!!  todo
                   # turn off terminations via limit violation and parameterize the reward function
                   'reward_function': gem.reward_functions.WeightedSumOfErrors(observed_states=['i_sq', 'i_sd'],
                                                                               reward_weights={'i_sq': 1, 'i_sd': 1},
                                                                               constraint_monitor=SqdCurrentMonitor(),
                                                                               gamma=gamma,
                                                                               reward_power=1
                                                                               ),

                   'reference_generator': rg,
                   # define a numerical solver of adequate accuracy
                   'ode_solver': 'euler'  # 'scipy.solve_ivp'
                   # 'state_filter': ['omega' , 'torque','i_sq', 'i_sd','u_sq', 'u_sd','epsilon']

                   }

# kwargs for evaluation environment:
gym_env_kwargs2 = {'visualization': MotorDashboard(plots=['i_sq', 'i_sd', 'reward']),
                   # parameterize the PMSM
                   'motor_parameter': motor_parameter,
                   'limit_values': limit_values,
                   'nominal_values': nominal_values,
                   'u_sup': u_sup,
                   'load': 'ConstSpeedLoad',  # ConstantSpeedLoad(omega_fixed=1000 * np.pi / 30),
                   'load_initializer': {'random_init': 'uniform', },
                   'tau': tau,
                   'motor_initializer': {'random_init': 'uniform', },

                   ## pass a reward function with a gamma!!  todo
                   # turn off terminations via limit violation and parameterize the reward function
                   'reward_function': gem.reward_functions.WeightedSumOfErrors(observed_states=['i_sq', 'i_sd'],
                                                                               reward_weights={'i_sq': 1, 'i_sd': 1},
                                                                               constraint_monitor=SqdCurrentMonitor(),
                                                                               gamma=gamma,
                                                                               reward_power=1
                                                                               ),

                   'reference_generator': rg,
                   # define a numerical solver of adequate accuracy
                   'ode_solver': 'euler'  # 'scipy.solve_ivp'
                   # 'state_filter': ['omega' , 'torque','i_sq', 'i_sd','u_sq', 'u_sd','epsilon']
                   }


class EpsilonWrapper(ObservationWrapper):
    """Changes Epsilon in a flattened observation to cos(epsilon) and sin(epsilon)"""

    def __init__(self, env, epsilon_idx):
        super(EpsilonWrapper, self).__init__(env)
        self.EPSILON_IDX = epsilon_idx
        new_low = np.concatenate((self.env.observation_space.low[:self.EPSILON_IDX], np.array([-1.]),
                                  self.env.observation_space.low[self.EPSILON_IDX:]))
        new_high = np.concatenate((self.env.observation_space.high[:self.EPSILON_IDX], np.array([1.]),
                                   self.env.observation_space.high[self.EPSILON_IDX:]))

        self.observation_space = Box(new_low, new_high)

    def observation(self, observation):
        cos_eps = np.cos(observation[self.EPSILON_IDX] * np.pi)
        sin_eps = np.sin(observation[self.EPSILON_IDX] * np.pi)
        observation = np.concatenate(
            (observation[:self.EPSILON_IDX], np.array([cos_eps, sin_eps]), observation[self.EPSILON_IDX + 1:]))
        return observation


class ShowProgress:
    def __init__(self, total):
        self.counter = 0
        self.total = total

    def __call__(self, trajectory):
        if not trajectory.is_boundary():
            self.counter += 1
        if self.counter % 100 == 0:
            print("\r{}/{}".format(self.counter, self.total), end="")


def compute_avg_return(environment, policy, num_steps=1000):
    total_return = 0.0
    steps = 0
    time_step = environment.reset()

    for _ in range(num_steps):

        action_step = policy.action(time_step)
        time_step = environment.step(action_step.action)

        total_return += time_step.reward
        steps += 1
        if time_step.is_last():
            time_step = environment.reset()

    avg_return = total_return
    return avg_return.numpy()[0]


env_name = "PMSMDisc-v1"
t_env = gem.make(env_name, **gym_env_kwargs1)  # define a PMSM with continuous action space
# eps_idx = t_env._physical_system.state_names.index('epsilon')
eps_idx = 7
t_env_f = EpsilonWrapper(FlattenObservation(t_env), eps_idx)

t_py_env = suite_gym.wrap_env(t_env_f, max_episode_steps=10000)
train_env = tf_py_environment.TFPyEnvironment(t_py_env)

ev_env = gem.make(env_name, **gym_env_kwargs2)  # define a PMSM with continuous action space
ev_env_f = EpsilonWrapper(FlattenObservation(ev_env), eps_idx)  # FlattenObservation(ev_env)

eval_py_env = suite_gym.wrap_env(ev_env_f, max_episode_steps=10000)
eval_tf_env = tf_py_environment.TFPyEnvironment(eval_py_env)

# Hyper-parameters
num_iterations = 500000

initial_collect_steps = 10000
collect_steps_per_iteration = 1
replay_buffer_max_length = 200000

batch_size = 25
learning_rate = 1e-4
eval_interval = 20000
fc_layer_params = (64, 64)

# create a neural network:

q_net = q_network.QNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    fc_layer_params=fc_layer_params)

# instantiate DQN agent

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
train_step_counter = tf.Variable(0)
# decay exploration for epsilon greedy:
global_step = tf.compat.v1.train.get_or_create_global_step()
start_epsilon = 0.7
n_of_steps = int(0.2 * num_iterations)
end_epsilon = 0.05
epsilon = tf.compat.v1.train.polynomial_decay(
    start_epsilon,
    train_step_counter,
    n_of_steps,
    end_learning_rate=end_epsilon)

agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    epsilon_greedy=epsilon,
    # target_update_tau=0.001,
    optimizer=optimizer,
    gamma=gamma,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter)

agent.initialize()

# create a replay buffer

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_max_length)

replay_buffer_observer = replay_buffer.add_batch

# Driver to take action steps in the environment


collect_driver = DynamicStepDriver(
    train_env,
    agent.collect_policy,
    observers=[replay_buffer_observer])


# fill the replay buffer initially with trajectories from a random policy

initial_collect_policy = RandomTFPolicy(train_env.time_step_spec(),
                                        train_env.action_spec())
init_driver = DynamicStepDriver(
    train_env,
    initial_collect_policy,
    observers=[replay_buffer.add_batch, ShowProgress(initial_collect_steps)],
    num_steps=initial_collect_steps)
final_time_step, final_policy_state = init_driver.run()

# dataset is sampled from the replay buffer

dataset = replay_buffer.as_dataset(
    num_parallel_calls=3,
    sample_batch_size=batch_size,
    num_steps=2).prefetch(3)

iterator = iter(dataset)



# Training loop for the agent
agent.train = common.function(agent.train)
collect_driver.run = function(collect_driver.run)

# Reset the train step
agent.train_step_counter.assign(0)

time_step = None
policy_state = agent.collect_policy.get_initial_state(train_env.batch_size)
# iterator = iter(dataset)
cum_reward = compute_avg_return(eval_tf_env, agent.policy, 2000)
returns = [cum_reward]
loss = []
start_time = time.time()
for iteration in range(num_iterations):
    time_step, policy_state = collect_driver.run(time_step, policy_state)
    trajectories, buffer_info = next(iterator)
    train_loss = agent.train(trajectories)
    loss.append(train_loss.loss.numpy())

    step = agent.train_step_counter.numpy()

    global_step.assign(step)
    #     if step % eval_interval ==0 :
    #         cum_reward = compute_avg_return(eval_tf_env, agent.policy, num_steps=2000)
    #         print('step = {0}: Cumulative Reward = {1}'.format(step, cum_reward))
    #         returns.append(cum_reward)

    # t_env_f.render()
    print("\r{} loss:{:.5f}".format(
        iteration, train_loss.loss.numpy()), end="")

end_time = time.time()
print("total training time is :", end_time - start_time)
#     if iteration % 1000 == 0:
#         log_metrics(train_metrics)


# Test the model for 1Million steps
step = 1000000
cum_reward = compute_avg_return(eval_tf_env, agent.policy, num_steps=step)
print('step = {0}: Cumulative Reward = {1}'.format(step, cum_reward))
