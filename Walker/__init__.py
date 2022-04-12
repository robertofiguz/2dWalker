from gym.envs.registration import register

register(
    id='Walker-v0',
    entry_point='Walker.envs:Walker',
)
