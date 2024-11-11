import numpy as np
import settings
import math

class Algorithm1:
    def __init__(self, env, time_horizon):
        self.env = env  # The environment instance
        self.time_horizon = time_horizon  # Total time horizon
        self.num_arms = env.get_num_arms()
        self.costs = env.get_costs()
        self.threshold = env.get_threshold()
        self.means_estimates = np.zeros(self.num_arms)  # Estimated means
        self.n_pulls = np.zeros(self.num_arms, dtype=int)  # Number of times each arm was pulled
        self.chosen_arm_index = None

    def run(self):
        # **Exploration phase:** Pull each arm once
        for i in range(self.num_arms):
            reward = self.env.pull_arm(i)
            self.n_pulls[i] += 1
            # Update the estimated mean
            self.means_estimates[i] += (reward - self.means_estimates[i]) / self.n_pulls[i]

        # Identify feasible arms with sample reward >= threshold
        feasible_arms = [i for i in range(self.num_arms) if self.means_estimates[i] >= self.threshold]

        if not feasible_arms:
            print("No arms have estimated mean reward above the threshold.")
            self.chosen_arm_index = None
            return

        # **Exploitation phase:**
        if len(feasible_arms) == self.num_arms:
            # If all arms have sample reward >= threshold, pull the least explored arm
            # Since all arms have been pulled once, choose the arm with the smallest index
            self.chosen_arm_index = feasible_arms[0]
        else:
            # Pull the cheapest arm among feasible arms
            self.chosen_arm_index = min(feasible_arms, key=lambda i: self.costs[i])

        # Exploitation phase: Pull the chosen arm for the remaining time
        remaining_time = self.time_horizon - self.num_arms  # We have pulled each arm once
        for _ in range(remaining_time):
            reward = self.env.pull_arm(self.chosen_arm_index)
            self.n_pulls[self.chosen_arm_index] += 1
            # Optionally update the mean estimate (not required per your instruction)
            # self.means_estimates[self.chosen_arm_index] += (reward - self.means_estimates[self.chosen_arm_index]) / self.n_pulls[self.chosen_arm_index]

    # Getter methods for reporting
    def get_estimated_means(self):
        return self.means_estimates

    def get_n_pulls(self):
        return self.n_pulls

    def get_chosen_arm(self):
        return self.chosen_arm_index

    


import numpy as np
import math

class Algorithm2:
    def __init__(self, env, time_horizon, hyperparameters, is_random_exploration=False, explore_limit=0):
        self.env = env  # The environment instance
        self.time_horizon = time_horizon  # Total time horizon
        self.is_random_exploration = is_random_exploration  # Hyperparameter
        self.hyperparameters = hyperparameters  # List of hyperparameters (weights for the features)
        self.num_arms = env.get_num_arms()
        self.total_number_of_arms = self.num_arms
        self.costs = env.get_costs()
        self.threshold = env.get_threshold()
        self.pulling_history = [[] for _ in range(self.num_arms)]  # List of rewards for each arm
        self.means_estimates = np.zeros(self.num_arms)  # Estimated means
        self.n_pulls = np.zeros(self.num_arms, dtype=int)  # Number of times each arm was pulled
        self.chosen_arm_index = None
        self.scores = []  # To store scores for each arm
        self.explore_limit = explore_limit
        
        self.arms_history = []  # List to store the sequence of arms selected
        self.rewards_history = []  # List to store the rewards obtained
        self.total_pulls_done = 0  # Keep track of total pulls
        
        


    def update_mean_estimate(self, arm_index, reward):
        # Efficiently update the mean estimate using incremental formula
        n = self.n_pulls[arm_index]
        if n == 1:
            self.means_estimates[arm_index] = reward
        else:
            old_mean = self.means_estimates[arm_index]
            new_mean = old_mean + (reward - old_mean) / n
            self.means_estimates[arm_index] = new_mean

    def run(self):
        # Exploration phase
        for i in range(self.num_arms):
            # Pull each arm once
            reward = self.env.pull_arm(i)
            self.n_pulls[i] += 1
            self.pulling_history[i].append(reward)
            self.update_mean_estimate(i, reward)

        total_exploration_pulls = int(np.ceil(self.time_horizon ** self.explore_limit)) - 1
        pulls_remaining = total_exploration_pulls - self.num_arms

        if self.is_random_exploration:
            # Random exploration: Pull arms randomly with equal probability
            for _ in range(pulls_remaining):
                arm_index = np.random.choice(self.num_arms)
                reward = self.env.pull_arm(arm_index)
                self.n_pulls[arm_index] += 1
                self.pulling_history[arm_index].append(reward)
                self.update_mean_estimate(arm_index, reward)
        else:
            # Sequential exploration: Pull each arm in order
            arm_sequence = list(range(self.num_arms))
            arm_index = 0
            while pulls_remaining > 0:
                # Pull the current arm
                current_arm = arm_sequence[arm_index]
                reward = self.env.pull_arm(current_arm)
                self.n_pulls[current_arm] += 1
                self.pulling_history[current_arm].append(reward)
                self.update_mean_estimate(current_arm, reward)
                pulls_remaining -= 1
                # Move to next arm
                arm_index = (arm_index + 1) % self.num_arms

        # Exploitation phase
        total_pulls_done = self.num_arms + total_exploration_pulls
        remaining_time = self.time_horizon - total_pulls_done

        for _ in range(remaining_time):
            # Update remaining_pulls
            remaining_pulls = self.time_horizon - total_pulls_done

            # Prepare features and calculate scores
            scores = []
            for i in range(self.num_arms):
                # Extract features for arm i
                sample_mean = self.means_estimates[i]
                #sample_variance = np.var(self.pulling_history[i], ddof=1) if self.n_pulls[i] > 1 else 0.0
                #sample_median = np.median(self.pulling_history[i])
                #sample_max = np.max(self.pulling_history[i])
                #sample_min = np.min(self.pulling_history[i])
                cost_i = self.costs[i]
                reward_threshold = self.threshold  # Same for all arms

                missing_reward = max(0, reward_threshold - sample_mean)
                missing_reward_percentage = missing_reward / reward_threshold if reward_threshold != 0 else 0.0
                missing_reward_percentage_log = math.log(missing_reward_percentage + 1)
                missing_reward_percentage_exp = math.exp(missing_reward_percentage)

                reward_overflow = max(0, sample_mean - reward_threshold)
                reward_overflow_percentage = reward_overflow / reward_threshold if reward_threshold != 0 else 0.0
                reward_overflow_percentage_log = math.log(reward_overflow_percentage + 1)
                reward_overflow_percentage_exp = math.exp(reward_overflow_percentage)

                pull_number = self.n_pulls[i]
                ucb = math.sqrt(4 * math.log(total_pulls_done + 1) / pull_number) if pull_number > 0 else float('inf')

                remaining_percentage_inv = remaining_pulls / pull_number if pull_number > 0 else 0.0
                remaining_percentage = 1/(remaining_pulls+1 / pull_number) if pull_number > 0 else 0.0

                # Initialize other features to zero
                other_cost1 = 0.0
                exist1 = 0
                sample_mean1 = 0.0
                missing_reward1 = 0.0
                missing_reward_percentage1 = 0.0
                missing_reward_percentage_log1 = 0.0
                missing_reward_percentage_exp1 = 0.0
                reward_overflow1 = 0.0
                reward_overflow_percentage1 = 0.0
                reward_overflow_percentage_log1 = 0.0
                reward_overflow_percentage_exp1 = 0.0
                pull_number1 = 0.0
                ucb1 = 0.0
                remaining_percentage1 = 0.0

                other_cost2 = 0.0
                exist2 = 0
                sample_mean2 = 0.0
                missing_reward2 = 0.0
                missing_reward_percentage2 = 0.0
                missing_reward_percentage_log2 = 0.0
                missing_reward_percentage_exp2 = 0.0
                reward_overflow2 = 0.0
                reward_overflow_percentage2 = 0.0
                reward_overflow_percentage_log2 = 0.0
                reward_overflow_percentage_exp2 = 0.0
                pull_number2 = 0.0
                ucb2 = 0.0
                remaining_percentage2 = 0.0
                
                in_set1 = 1 if sample_mean >= reward_threshold else 0
                in_set2 = 1 if sample_mean + ucb >= reward_threshold else 0

                # **Compute other_cost1 and associated features**
                # Find the cheapest other arm (excluding current arm i) with sample_mean >= threshold
                '''
                feasible_arms1 = [
                    j for j in range(self.num_arms)
                    if j != i and self.means_estimates[j] >= reward_threshold
                ]
                if feasible_arms1:
                    exist1 = 1
                    # Find the arm with the lowest cost among feasible arms
                    cheapest_arm1 = min(feasible_arms1, key=lambda j: self.costs[j])
                    other_cost1 = self.costs[cheapest_arm1]
                    # Extract features for arm cheapest_arm1
                    sample_mean1 = self.means_estimates[cheapest_arm1]
                    missing_reward1 = max(0, reward_threshold - sample_mean1)
                    missing_reward_percentage1 = missing_reward1 / reward_threshold if reward_threshold != 0 else 0.0
                    missing_reward_percentage_log1 = math.log(missing_reward_percentage1 + 1)
                    missing_reward_percentage_exp1 = math.exp(missing_reward_percentage1)
                    reward_overflow1 = max(0, sample_mean1 - reward_threshold)
                    reward_overflow_percentage1 = reward_overflow1 / reward_threshold if reward_threshold != 0 else 0.0
                    reward_overflow_percentage_log1 = math.log(reward_overflow_percentage1 + 1)
                    reward_overflow_percentage_exp1 = math.exp(reward_overflow_percentage1)
                    pull_number1 = self.n_pulls[cheapest_arm1]
                    ucb1 = math.sqrt(4 * math.log(total_pulls_done + 1) / pull_number1) if pull_number1 > 0 else 0
                    remaining_percentage1_inv = remaining_pulls / pull_number1 if pull_number1 > 0 else 0.0
                    remaining_percentage1 = 1/(remaining_pulls+1 / pull_number1) if pull_number1 > 0 else 0.0

                # **Compute other_cost2 and associated features**
                # Find the cheapest other arm (excluding current arm i) with sample_mean + ucb >= threshold
                feasible_arms2 = []
                for j in range(self.num_arms):
                    if j != i:
                        sample_mean_j = self.means_estimates[j]
                        pull_number_j = self.n_pulls[j]
                        ucb_j = math.sqrt(4 * math.log(total_pulls_done + 1) / pull_number_j) if pull_number_j > 0 else 0
                        if sample_mean_j + ucb_j >= reward_threshold:
                            feasible_arms2.append(j)

                if feasible_arms2:
                    exist2 = 1
                    # Find the arm with the lowest cost among feasible arms
                    cheapest_arm2 = min(feasible_arms2, key=lambda j: self.costs[j])
                    other_cost2 = self.costs[cheapest_arm2]
                    # Extract features for arm cheapest_arm2
                    sample_mean2 = self.means_estimates[cheapest_arm2]
                    missing_reward2 = max(0, reward_threshold - sample_mean2)
                    missing_reward_percentage2 = missing_reward2 / reward_threshold if reward_threshold != 0 else 0.0
                    missing_reward_percentage_log2 = math.log(missing_reward_percentage2 + 1)
                    missing_reward_percentage_exp2 = math.exp(missing_reward_percentage2)
                    reward_overflow2 = max(0, sample_mean2 - reward_threshold)
                    reward_overflow_percentage2 = reward_overflow2 / reward_threshold if reward_threshold != 0 else 0.0
                    reward_overflow_percentage_log2 = math.log(reward_overflow_percentage2 + 1)
                    reward_overflow_percentage_exp2 = math.exp(reward_overflow_percentage2)
                    pull_number2 = self.n_pulls[cheapest_arm2]
                    ucb2 = math.sqrt(4 * math.log(total_pulls_done + 1) / pull_number2) if pull_number2 > 0 else 0
                    remaining_percentage2_inv = remaining_pulls / pull_number2 if pull_number2 > 0 else 0.0
                    remaining_percentage2 = 1/(remaining_pulls+1 / pull_number2) if pull_number2 > 0 else 0.0
                '''

                zero = 0  # Placeholder if needed

                # Create the feature vector
                features_i = [
                    
                    sample_mean,
                    #sample_variance,
                    #sample_median,
                    #sample_max,
                    #sample_min,
                    cost_i,

                    #missing_reward,
                    missing_reward_percentage,
                    #missing_reward_percentage_log,
                    #missing_reward_percentage_exp,
                    #reward_overflow,
                    reward_overflow_percentage,
                    #reward_overflow_percentage_log,
                    #reward_overflow_percentage_exp,

                    pull_number,
                    ucb,

                    remaining_pulls,
                    #remaining_percentage,
                    #remaining_percentage_inv,


                    
                    
                    in_set1,
                    in_set2

                    
                ]

                # Calculate the score for arm i
                score_i = np.dot(self.hyperparameters, features_i)
                scores.append(score_i)

            # Store the scores for reporting if needed
            self.scores = scores

            # Select the arm with the highest score
            self.chosen_arm_index = np.argmax(scores)

            # Pull the chosen arm
            reward = self.env.pull_arm(self.chosen_arm_index)
            self.n_pulls[self.chosen_arm_index] += 1
            self.pulling_history[self.chosen_arm_index].append(reward)
            self.update_mean_estimate(self.chosen_arm_index, reward)

            # Record history
            self.arms_history.append(self.chosen_arm_index)
            self.rewards_history.append(reward)

            # Update total pulls done
            total_pulls_done += 1
            self.total_pulls_done = total_pulls_done

    # Getter methods for reporting
    def get_estimated_means(self):
        return self.means_estimates

    def get_n_pulls(self):
        return self.n_pulls

    def get_chosen_arm(self):
        return self.chosen_arm_index

    def get_scores(self):
        return self.scores
    
    
    
    
    def get_arms_history(self):
        return self.arms_history

    def get_rewards_history(self):
        return self.rewards_history


class Algorithm3:
    def __init__(self, env, time_horizon):
        self.env = env  # The environment instance
        self.time_horizon = time_horizon  # Total time horizon
        self.num_arms = env.get_num_arms()
        self.costs = env.get_costs()
        self.threshold = env.get_threshold()
        self.means_estimates = np.zeros(self.num_arms)  # Estimated means
        self.n_pulls = np.zeros(self.num_arms, dtype=int)  # Number of times each arm was pulled
        self.arms_history = []  # List of arms pulled
        self.rewards_history = []  # List of rewards obtained
        self.total_pulls_done = 0

    def run(self):
        T = self.time_horizon
        K = self.num_arms

        # **Exploration phase:** Pull each arm once
        for i in range(K):
            reward = self.env.pull_arm(i)
            self.n_pulls[i] += 1
            self.means_estimates[i] = reward  # Since pulled once
            self.arms_history.append(i)
            self.rewards_history.append(reward)
            self.total_pulls_done += 1

        # **Exploitation phase:**
        for t in range(K, T):
            # For each arm, compute empirical mean and UCB
            ucb_values = np.zeros(K)
            for i in range(K):
                n_i = self.n_pulls[i]
                mu_i = self.means_estimates[i]
                confidence = math.sqrt((2 * math.log(T)) / n_i)
                ucb_i = mu_i + confidence
                ucb_values[i] = ucb_i

            # Construct Feasible Set: Arms with UCB >= threshold
            feasible_arms = [i for i in range(K) if ucb_values[i] >= self.threshold]

            if feasible_arms:
                # Select the cheapest arm among feasible arms
                It = min(feasible_arms, key=lambda i: self.costs[i])
            else:
                # If no feasible arms, select the arm with the highest UCB
                It = np.argmax(ucb_values)

            # Pull arm It
            reward = self.env.pull_arm(It)
            self.n_pulls[It] += 1
            # Update mean estimate using incremental formula
            n = self.n_pulls[It]
            old_mean = self.means_estimates[It]
            self.means_estimates[It] += (reward - old_mean) / n

            # Record history
            self.arms_history.append(It)
            self.rewards_history.append(reward)
            self.total_pulls_done += 1

    # Getter methods for reporting
    def get_estimated_means(self):
        return self.means_estimates

    def get_n_pulls(self):
        return self.n_pulls

    def get_arms_history(self):
        return self.arms_history

    def get_rewards_history(self):
        return self.rewards_history
