from gymnasium.envs.toy_text.blackjack import draw_card, draw_hand, sum_hand, usable_ace
import numpy as np


######################
## Frozen Lake Support
######################

# get 2d s function
def get_s_frozen_lake(env):
    return env.unwrapped.s
    
# set_s
def set_s_frozen_lake(env, s):
    env.unwrapped.s = s
    return True

# get state size
def get_s_size_frozen_lake():
    return 64


####################
## Blackjack Support
####################

# get blackjack
def get_s_blackjack(env):
    # https://stackoverflow.com/questions/7367770/how-to-flatten-or-index-3d-array-in-1d-array
    x, y, z = tuple(int(v) for v in env.unwrapped._get_obs())
    x_dim, y_dim, z_dim = tuple(int(v.n) for v in env.observation_space)
    # https://coderwall.com/p/fzni3g/bidirectional-translation-between-1d-and-3d-arrays
    i = x + y * x_dim + z * x_dim * y_dim
    return i

# set blackjack
def set_s_blackjack(env, s):
    # Returns True is valid state, or False if not (if False, state was not set)
    # relevant source code from here:
    # https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/toy_text/blackjack.py
    # This function must work exactly like the source code under the hood
    # TODO reverse the index calculation, set result to new_state
    # https://coderwall.com/p/fzni3g/bidirectional-translation-between-1d-and-3d-arrays
    # https://stackoverflow.com/questions/11316490/convert-a-1d-array-index-to-a-3d-array-index
    x_dim, y_dim, z_dim = tuple(int(v.n) for v in env.observation_space)
    x, y, z = (
        s % x_dim,
        (s // x_dim) % y_dim,
        s // (x_dim * y_dim)
    )
    # new_state = (0, 0, 0)
    # env.unwrapped.player = [0, new_state[0]]
    player = draw_hand(np.random)
    i = 0
    while (sum_hand(player) != x) or (usable_ace(player) != z):
        player = draw_hand(np.random)
        i += 1
        if i > 10000:
            return False
    env.unwrapped.player = player
    # Set dealer hand
    env.unwrapped.dealer = [
        y, 
        draw_card(np.random)
        ]
    return True

def get_s_size_blackjack():
    return 32 * 11 * 2


def get_blackjack_win_ratio(env, Pi):
    np.random.seed(42)
    results = list()
    for _ in range(100000):
        env.reset()
        terminated = False
        while not terminated:
            a = Pi[get_s_blackjack(env)]
            s, r, terminated, truncated, _ = env.step(a)
        results.append(r)
    win_ratio = (np.array(results) == 1).mean()
    print("win ratio: ", win_ratio, " hit hands: ", Pi.sum())
    return win_ratio

def get_frozen_lake_win_ratio(env, Pi):
    np.random.seed(42)
    results = list()
    length = list()
    for _ in range(1000):
        env.reset()
        terminated = False
        i=0
        while not terminated:
            i+=1
            a = Pi[get_s_frozen_lake(env)]
            s, r, terminated, truncated, _ = env.step(a)
        results.append(r)
        length.append(i)
    win_ratio = (np.array(results) == 1).mean()
    avg_length = np.mean(length)
    print("win ratio: ", win_ratio, " avg len: %.1f" % avg_length)
    return win_ratio, avg_length