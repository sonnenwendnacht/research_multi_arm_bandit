import numpy as np

class Environment:
    MAX_NUM = float('inf')  # Represents an infinitely large number

    def __init__(self, filename=None):
        self.num_arms = None
        self.costs = None
        self.means = None
        self.threshold = None
        self.time = 0
        self.total_time = 0
        self.arm_pulls = []
        self.arm_rewards = []
        self.history = []
        self.random_seed = None  # Optional: Seed for reproducibility

        if filename:
            self.load_settings(filename)
        else:
            # Default settings if no file is provided
            self.threshold = 0.5
            self.num_arms = 5
            self.costs = [1.0, 2.0, 0.5, 1.5, 1.0]
            self.means = [0.4, 0.6, 0.55, 0.45, 0.5]

        # Initialize arm pulls and rewards after settings are loaded
        self.arm_pulls = [0] * self.num_arms
        self.arm_rewards = [0.0] * self.num_arms

        # Set random seed if specified
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

    def load_settings(self, filename):
        with open(filename, 'r') as f:
            code = f.read()
            # Execute the file code in the context of the Environment instance
            exec(code, {}, self.__dict__)

    def pull_arm(self, arm_index):
        """
        Simulate pulling an arm and return a reward sample.
        The reward distribution can be customized per arm if specified.
        """
        mean = self.means[arm_index]
        # Check if a custom distribution is specified for this arm
        if hasattr(self, 'arm_distributions') and self.arm_distributions.get(arm_index):
            dist_info = self.arm_distributions[arm_index]
            dist_name = dist_info['name']
            params = dist_info['params']

            # Generate reward based on the specified distribution
            if dist_name == 'gaussian':
                reward = np.random.normal(loc=params['mean'], scale=params['std'])
            elif dist_name == 'bernoulli':
                reward = np.random.binomial(n=1, p=params['p'])
            # Add other distributions as needed
            else:
                # Default to Gaussian if unrecognized
                reward = np.random.normal(loc=mean, scale=1)
        else:
            # Default to Gaussian distribution with variance 1
            reward = np.random.normal(loc=mean, scale=1)

        # Update internal state
        self.time += 1
        self.total_time += 1
        self.arm_pulls[arm_index] += 1
        self.arm_rewards[arm_index] += reward
        self.history.append({'arm': arm_index, 'reward': reward})
        return reward

    def calculate_regret(self):
        # Identify feasible arms based on true means
        feasible_arms = [i for i in range(self.num_arms) if self.means[i] >= self.threshold]
        if feasible_arms:
            x = min([self.costs[i] for i in feasible_arms])  # Cost of cheapest feasible arm
        else:
            x = self.MAX_NUM  # No feasible arms
            print("No feasible arms with expected reward >= threshold for cost regret calculation.")

        total_quality_regret = 0.0
        total_cost_regret = 0.0

        # Calculate regret for each pull
        for event in self.history:
            arm_index = event['arm']
            expected_return = self.means[arm_index]  # True mean reward
            step_quality_regret = max(0, self.threshold - expected_return)
            total_quality_regret += step_quality_regret
            cost_i = self.costs[arm_index]
            step_cost_regret = max(0, cost_i - x)
            total_cost_regret += step_cost_regret

        self.history = []  # Reset history
        return total_quality_regret, total_cost_regret

    # Getter methods
    def get_num_arms(self):
        return self.num_arms

    def get_costs(self):
        return self.costs

    def get_threshold(self):
        return self.threshold

    def get_arm_pulls(self):
        return self.arm_pulls

    def get_estimated_means(self):
        return [
            self.arm_rewards[i] / self.arm_pulls[i] if self.arm_pulls[i] > 0 else 0.0
            for i in range(self.num_arms)
        ]
