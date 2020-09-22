import sys, os

from pathlib import Path
import sys
sys.path.append(str(Path().resolve().parent.parent))
sys.path.append(os.path.abspath(os.path.join('..')))
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from setting_environment import set_env

# Training parameters.
gamma = 0.99
time_limit = True
tau = 1e-5
simulation_time = 5  # seconds
buffer_size = 200000  # number of old obersation steps saved
learning_starts = 10000  # memory warmup
# train_freq = 1 # prediction network gets an update each train_freq's step
batch_size = 25  # mini batch size drawn at each update step
env = set_env(time_limit, gamma)
nb_actions = env.action_space.n
window_length = 1
model = Sequential()
model.add(Flatten(input_shape=(window_length,) + env.observation_space.shape))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(nb_actions, activation='linear'))

memory = SequentialMemory(limit=200000, window_length=window_length)
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(eps=0.2), 'eps', 1, 0.05, 0, 50000)

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
        visualize=False,
        nb_max_episode_steps=10000,
        log_interval=1000
        )

dqn.save_weights('save_dqn_keras.hdf5', overwrite=True)

model.save_weights('save_model_keras.hdf5')