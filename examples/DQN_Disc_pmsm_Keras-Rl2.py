from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from gym.wrappers import FlattenObservation
from gym.spaces import Discrete

import sys, os
import numpy as np

sys.path.append(os.path.abspath(os.path.join('..')))
import gym_electric_motor as gem
from gym_electric_motor.reward_functions import WeightedSumOfErrors
from gym_electric_motor.visualization import MotorDashboard
from gym_electric_motor.reference_generators import WienerProcessReferenceGenerator
from gym_electric_motor.reference_generators import MultipleReferenceGenerator

gamma = 0.99
tau = 1e-5
simulation_time = 5  # seconds
buffer_size = 200000  # number of old obersation steps saved
learning_starts = 10000  # memory warmup
train_freq = 1  # prediction network gets an update each train_freq's step
batch_size = 25  # mini batch size drawn at each update step


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
                       )
u_sup = 350
nominal_values = dict(omega=4000 * 2 * np.pi / 60,
                      i=230,
                      u=u_sup
                      )

limit_values = nominal_values.copy()

q_generator = WienerProcessReferenceGenerator(reference_state='i_sq')
d_generator = WienerProcessReferenceGenerator(reference_state='i_sd')
rg = MultipleReferenceGenerator([q_generator, d_generator])

env = gem.make(
    # define a PMSM with discrete action space
    "PMSMDisc-v1",

    # parameterize the PMSM
    motor_parameter=motor_parameter,

    # Parameters to be visualised
    visualization=MotorDashboard(plots=['i_sq', 'i_sd', 'reward']),

    # update the limitations of the state space
    limit_values=limit_values,
    nominal_values=nominal_values,

    # set DC link voltage
    u_sup=u_sup,

    # define the speed at which the motor is operated - should be drawn randomly at each episode
    load='ConstSpeedLoad',
    load_initializer={'random_init': 'uniform', },

    # random motor parameters each episode
    motor_initializer={'random_init': 'uniform', },

    # define the duration of one sampling step
    tau=tau,

    # turn off terminations via limit violation and parameterize the reward function
    reward_function=WeightedSumOfErrors(observed_states=['i_sq', 'i_sd'],
                                        reward_weights={'i_sq': 1, 'i_sd': 1},
                                        constraint_monitor=SqdCurrentMonitor(),
                                        gamma=gamma,
                                        reward_power=1
                                        ),
    # reference_generator
    reference_generator=rg,

    # Numerical Solver
    ode_solver='euler',
)
env = FlattenObservation(env)
env.action_space = Discrete(7)
nb_actions = env.action_space.n
window_length = 1

model = Sequential()
model.add(Flatten(input_shape=(window_length,) + env.observation_space.shape))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(nb_actions, activation='linear'))

memory = SequentialMemory(limit=200000, window_length=window_length)
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(eps=0.2), 'eps', 1, 0.05, 0, 100000)

dqn = DQNAgent(
    model=model,
    policy=policy,
    nb_actions=nb_actions,
    memory=memory,
    gamma=gamma,
    batch_size=25,
    # train_interval=1,
    # memory_interval=1,
    target_model_update=1000,
    nb_steps_warmup=10000,
    enable_double_dqn=True
)

dqn.compile(Adam(lr=1e-4),
            metrics=['mse']
            )

dqn.fit(env,
        nb_steps=500000,
        action_repetition=1,
        verbose=2,
        visualize=True,
        nb_max_episode_steps=10000,
        log_interval=1000
        )

dqn.test(env,
         nb_episodes=3,
         nb_max_episode_steps=100000,
         visualize=True
         )
