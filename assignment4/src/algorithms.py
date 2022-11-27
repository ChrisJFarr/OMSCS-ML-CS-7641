import numpy as np

from typing import Callable
from joblib import Parallel, delayed
from multiprocessing import cpu_count


def get_P_and_R(env, n_states, sample_size, set_s: Callable):
    shape = (n_states, env.action_space.n, n_states) 
    # Initialize P and R to 0's
    P = np.zeros(shape=shape, dtype=np.float32)
    R = np.zeros(shape=shape, dtype=np.float32)
    for j in range(n_states):
        for a in range(env.action_space.n):
            for _ in range(sample_size):
                # Setup environment_
                env.reset()
                is_state = set_s(env, j)
                if not is_state:
                    break
                # Take the action
                k, r, terminated, truncated, _ = env.step(a)
                # Update R
                R[j, a, k]=r
                # Update P
                P[j, a, k]+=1/sample_size
    return P, R
    

# Policy Iteration Implementation
def policy_iteration(env, set_s: Callable, n_states, 
    gamma=1.0, max_delta = 0.00001, max_iter = 100, sample_size=100, P=None, R=None):  #  get_s: Callable,, s_size: Callable
    # Build transition and reward matrices
    if P is None or R is None:
        P, R = get_P_and_R(env, n_states, sample_size, set_s)

    # Initialization
    V = np.zeros(shape=(n_states, ), dtype=np.float32)
    Pi = np.zeros(shape=(n_states, ), dtype=np.int32)

    def evaluate(max_delta):
        # Policy Evaluation
        delta = np.inf
        i_ = 0
        while delta > max_delta:
            i_+=1
            delta = 0
            for s in range(n_states):
                v = V[s]
                V[s] = np.sum(P[s, Pi[s], :] * (R[s, Pi[s], :] + gamma * V))
                delta = max(delta, abs(v - V[s]))
            if i_>max_iter:
                break
        # print("iters:", i)
        return delta

    def improve():
        # Policy Improvement
        policy_stable = True
        for s in range(n_states):
            b = Pi[s]
            V_discounted = gamma * V
            Pi[s] = np.argmax([sum(P[s, a, :] * (R[s, a, :] + V_discounted))
                for a in range(env.action_space.n)])
            if b != Pi[s]:
                policy_stable = False
                # print("not stable")
        return policy_stable


    # Policy Iteration
    i = 0
    while True:
        delta = evaluate(max_delta)
        # print("delta:", delta)
        policy_stable = improve()
        if policy_stable:
            break
        i += 1
        if i == max_iter:
            print("max iters reached")
            break
    print("iterations: ", i)

    return P, R, V, Pi


# Value Iteration
def value_iteration(env, set_s: Callable, n_states, 
    gamma=1.0, max_delta = 0.00001, max_iter = 100, sample_size=100, P=None, R=None):

    if P is None or R is None:
        P, R = get_P_and_R(env, n_states, sample_size, set_s)

    # Initialization
    V = np.zeros(shape=(n_states, ), dtype=np.float32)
    Pi = np.zeros(shape=(n_states, ), dtype=np.int32)

    # Policy Evaluation
    delta = np.inf
    i = 0
    while delta > max_delta:
        delta = 0
        for s in range(n_states):
            v = V[s]
            V[s] = np.max([np.sum(P[s, a, :] * (R[s, a, :] + gamma * V)) 
                for a in range(env.action_space.n)])
            delta = max(delta, abs(v - V[s]))
        if i>=max_iter:
            print("max iterations reached...")
            break
        i+=1
    print("iterations: ", i)
    # Compute policy
    for s in range(n_states):
        b = Pi[s]
        V_discounted = gamma * V
        Pi[s] = np.argmax([sum(P[s, a, :] * (R[s, a, :] + V_discounted))
            for a in range(env.action_space.n)])
    return P, R, V, Pi


# Q-learning

def update_q_table(Q, s, a, s_prime, r, alpha, gamma):
    new_val = ((1-alpha) * Q[s, a]) + alpha * (r + gamma * np.max(Q[s_prime, :]))
    if np.inf == new_val:
        print("inf", Q, s, a, s_prime, r, alpha, gamma)
        raise Exception
    Q[s, a] = new_val
    return


def q_learning_w_dyna(env, get_s: Callable, state_size: int, n_episodes = 100_000, epsilon = 1.0, epsilon_decay = 0.99999,
    min_epsilon = 0.0, dyna = None, dyna_period = None, alpha=0.1, gamma=1.0, reward_shaping=None, penalize_step=False, verbose=False):
    action_size = env.action_space.n
    Q = np.zeros(shape=(state_size, action_size), dtype=np.float64)
    successful_episodes = 0
    # Create matrix for neutral reward shaping if not passed
    if reward_shaping is None:
        reward_shaping = np.zeros(shape=(state_size, ), dtype=np.float64)
    experience = list()
    for i in range(n_episodes):
        obs, info = env.reset()
        s = get_s(env)
        terminated = False
        while not terminated:
            a = np.argmax(Q[s, :])
            if np.random.uniform(0, 1) < epsilon:
                a = np.random.choice(env.action_space.n)
            _, r, terminated, _, _ = env.step(a)
            if r > 0:
                successful_episodes+=1
            s_prime = get_s(env)
            if penalize_step:
                r -= 1
            r += reward_shaping[s_prime]
            if dyna is not None:
                if len(experience) > n_episodes:
                    experience.pop(0)
                experience.append((s, a, s_prime, r))
            else:
                # print(s, a, s_prime, r)
                update_q_table(Q, s, a, s_prime, r, alpha, gamma)
            s = s_prime
        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        if dyna is not None:
            if i % dyna_period == 0:
                # print("epsilon: ", epsilon)
                # perform dyna update 
                for _ in range(dyna):
                    experience_i = np.random.randint(0, len(experience))
                    update_q_table(Q, *experience[experience_i], alpha, gamma)
    print("final epsilon: %.5f" % epsilon, " successful episodes: %s/%s" % (successful_episodes, n_episodes))
    Pi = np.argmax(Q, axis=1)
    return Q, Pi