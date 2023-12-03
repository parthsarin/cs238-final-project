import pandas as pd
from collections import defaultdict
from tqdm import tqdm

def infer_fields(df):
    c = df.columns
    return {
        "o": [c for c in c if c.startswith("o_")],
        "a": [c for c in c if c.startswith("a_")],
        "r": [c for c in c if c.startswith("r")],
        "op": [c for c in c if c.startswith("op_")],
        "ap": [c for c in c if c.startswith("ap_")],
    }

def QMDP(df, k_max=50, gamma=0.95):
    fields = infer_fields(df)
    Q = defaultdict(float)  # Initialize Q-values

    # Simplified state and action extraction
    states = df[fields['o']].drop_duplicates().to_numpy()
    actions = df[fields['a']].drop_duplicates().to_numpy()

    # Simplified transition probability estimation
    def estimate_transition_probabilities(df, current_state, next_state):
        # Very basic estimation: frequency of direct transition
        total_transitions = len(df)
        specific_transitions = len(df[(df[fields['o']] == current_state) & (df[fields['op']] == next_state)])
        return specific_transitions / total_transitions if total_transitions > 0 else 0

    def update(Q, df, gamma):
        for state in states:
            state_tuple = tuple(state)  # Convert state array to tuple
            for action in actions:
                action_tuple = tuple(action)  # Convert action array to tuple
                # Calculate reward for this state-action pair
                reward = df[(df[fields['o']] == state_tuple) & (df[fields['a']] == action_tuple)]['r'].mean()

                # Calculate expected utility
                expected_utility = 0
                for next_state in states:
                    next_state_tuple = tuple(next_state)  # Convert next state array to tuple
                    transition_probability = estimate_transition_probabilities(df, state_tuple, next_state_tuple)
                    max_q_value = max(Q[next_state_tuple, a_tuple] for a_tuple in (tuple(a) for a in actions))
                    expected_utility += transition_probability * max_q_value

                # Update Q-value
                Q[state_tuple, action_tuple] = reward + gamma * expected_utility

        return Q

    # Run alpha vector iteration
    for _ in tqdm(range(k_max)):
        Q = update(Q, df, gamma)

    return Q

# Usage of the function
# df = pd.read_csv('your_dataframe.csv')  # Load your DataFrame here
# Q_values = QMDP(df, k_max=100, gamma=0.95)
