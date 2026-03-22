import numpy as np

def mc_policy_evaluation(episodes, gamma, n_states):
    V = np.zeros(n_states)
    returns_sum = np.zeros(n_states)
    returns_count = np.zeros(n_states)

    for episode in episodes:
        states, rewards = zip(*episode)
        T = len(states)

        # Precompute returns G_t for all t
        G = np.zeros(T)
        G[-1] = rewards[-1]
        for t in reversed(range(T - 1)):
            G[t] = rewards[t] + gamma * G[t + 1]

        visited = set()

        # IMPORTANT: iterate FORWARD for first-visit
        for t in range(T):
            s = states[t]
            if s not in visited:
                visited.add(s)
                returns_sum[s] += G[t]
                returns_count[s] += 1
                V[s] = returns_sum[s] / returns_count[s]

    return V