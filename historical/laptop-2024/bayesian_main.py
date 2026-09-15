import sys
import numpy as np
import settings
import algorithm  # Ensure this imports Algorithm1, Algorithm2, and Algorithm3
import time
import os
import pickle  # For saving and loading best hyperparameters
import re
import matplotlib.pyplot as plt  # Added for plotting
import random  # Use standard random module for consistent data types



# Add these imports for Bayesian Optimization
from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args
import json  # For saving query history
import warnings
warnings.filterwarnings("ignore")  # Suppress warnings for cleaner output

def main():
    # Initialize variables
    T = 50000  # Adjusted time horizon for demonstration purposes
    lambda_reg = 0.0001  # Regularization parameter

    feature_names = [
        "sample_mean",
        #"sample_variance",
        #"sample_median",
        #"sample_max",
        #"sample_min",
        "cost_i",
        #"missing_reward",
        "missing_reward_percentage",
        #"missing_reward_percentage_log",
        #"missing_reward_percentage_exp",
        #"reward_overflow",
        "reward_overflow_percentage",
        #"reward_overflow_percentage_log",
        #"reward_overflow_percentage_exp",
        "pull_number",
        "ucb",
        "remaining_pulls",
        #"remaining_percentage",
        #"remaining_percentage_inv",
        "in_set1",
        "in_set2"
    ]

    # Check for command-line arguments
    if len(sys.argv) < 2:
        print("Usage: python3 bayesian_main.py [filename]")
        sys.exit()

    # Get the filename from command-line arguments
    filename = sys.argv[1]

    # Extract 'x' from 'file_x' or 'file_x.txt'
    match = re.search(r'file_(\d+)', filename)
    if match:
        x = match.group(1)
    else:
        print("Invalid filename format. Expected 'file_x' or 'file_x.txt'")
        sys.exit()

    # Modify the save file name to include 'x'
    save_file = f'best_hyperparameters_{x}.pkl'

    # Prompt the user to select the algorithm or delete saved data
    algo_choice = input("Enter which algorithm to run (1, 2, or 3), or type 'DELETE' to remove saved hyperparameters: ").strip()

    if algo_choice == 'DELETE':
        confirmation = input(f"Are you sure you want to delete the saved best hyperparameters and query history for file_{x}? Type 'CONTINUE DELETE' to proceed: ").strip()
        if confirmation == 'CONTINUE DELETE':
            if os.path.exists(save_file):
                os.remove(save_file)
                print(f"Saved best hyperparameters for file_{x} have been deleted.")
            else:
                print(f"No saved best hyperparameters file found for file_{x}.")
            # Delete query history
            history_file = f'query_history_file_{x}.json'
            if os.path.exists(history_file):
                os.remove(history_file)
                print(f"Query history for file_{x} has been deleted.")
            else:
                print(f"No query history file found for file_{x}.")
        else:
            print("Deletion cancelled.")
        sys.exit()

    elif algo_choice not in ['1', '2', '3']:
        print("Invalid choice. Exiting.")
        sys.exit()

    if algo_choice == '1':
        # Code for Algorithm1 remains the same
        # Initialize the environment
        env = settings.Environment(filename=filename)

        # Initialize and run Algorithm1
        alg = algorithm.Algorithm1(env=env, time_horizon=T)
        alg.run()

        # After the algorithm has run, calculate regrets
        quality_regret, cost_regret = env.calculate_regret()

        # Output results
        print(f"\nAlgorithm1 selected arm: {alg.get_chosen_arm()}")
        print(f"Estimated means: {alg.get_estimated_means()}")
        print(f"Number of pulls per arm: {alg.get_n_pulls()}")
        print(f"Total quality regret: {quality_regret}")
        print(f"Total cost regret: {cost_regret}")

        plot_cost_regret(env, alg, algo_choice, T)

    elif algo_choice == '3':
        # Code for Algorithm3 remains the same
        # Initialize the environment
        env = settings.Environment(filename=filename)

        # Initialize and run Algorithm3
        alg = algorithm.Algorithm3(env=env, time_horizon=T)
        alg.run()

        # After the algorithm has run, calculate regrets
        quality_regret, cost_regret = env.calculate_regret()

        # Output results
        print(f"\nAlgorithm3 (UCB) selected arms:")
        for t, arm in enumerate(alg.get_arms_history(), 1):
            print(f"Time {t}: Pulled arm {arm}, Reward: {alg.get_rewards_history()[t-1]}")

        print(f"\nEstimated means: {alg.get_estimated_means()}")
        print(f"Number of pulls per arm: {alg.get_n_pulls()}")
        print(f"Total quality regret: {quality_regret}")
        print(f"Total cost regret: {cost_regret}")

        plot_cost_regret(env, alg, algo_choice, T)

    elif algo_choice == '2':
        # Prompt the user for the desired action
        print("Options for Algorithm2:")
        print("1: Run with the best saved hyperparameters.")
        print("2: Input custom hyperparameters.")
        print("3: Perform hyperparameter tuning.")
        action_choice = input("Enter your choice (1, 2, or 3): ").strip()

        if action_choice == '1':
            # Run with the best saved hyperparameters
            if os.path.exists(save_file):
                with open(save_file, 'rb') as f:
                    saved_data = pickle.load(f)
                    best_hyperparameters = saved_data['hyperparameters']
                    best_explore_limit = saved_data['explore_limit']
                    best_is_random_exploration = saved_data['is_random_exploration']
                    best_loss = saved_data['loss']
                    print("Loaded best saved hyperparameters.")
                # Initialize the environment
                env = settings.Environment(filename=filename)
                # Run Algorithm2 and display detailed outputs
                run_algorithm2_and_display_results(env, T, best_hyperparameters, best_explore_limit, best_is_random_exploration, save_file, lambda_reg, feature_names, is_custom_run=False, filename=filename)
            else:
                print("No saved best hyperparameters found. Please perform hyperparameter tuning first.")
                sys.exit()

        elif action_choice == '2':
            # Input custom hyperparameters
            print("Please input custom hyperparameters for Algorithm2.")
            hyperparameters = []
            for feature in feature_names:
                while True:
                    try:
                        value = float(input(f"Enter weight for {feature}: "))
                        hyperparameters.append(value)
                        break
                    except ValueError:
                        print("Invalid input. Please enter a numeric value.")

            # Input explore_limit
            while True:
                try:
                    explore_limit = float(input("Enter explore limit (e.g., 0.5): "))
                    break
                except ValueError:
                    print("Invalid input. Please enter a numeric value.")

            # Input is_random_exploration
            is_random_exploration_input = input("Use random exploration? (yes/no): ").strip().lower()
            is_random_exploration = is_random_exploration_input in ['yes', 'y']

            # Initialize the environment
            env = settings.Environment(filename=filename)
            # Run Algorithm2 and display detailed outputs
            run_algorithm2_and_display_results(env, T, hyperparameters, explore_limit, is_random_exploration, save_file, lambda_reg, feature_names, is_custom_run=True, filename=filename)
        elif action_choice == '3':
            # Perform hyperparameter tuning
            perform_bayesian_hyperparameter_tuning(filename, T, lambda_reg, feature_names, save_file)
        else:
            print("Invalid choice. Exiting.")
            sys.exit()

def run_algorithm2_and_display_results(env, T, hyperparameters, explore_limit, is_random_exploration, save_file, lambda_reg, feature_names, is_custom_run, filename):
    # Initialize and run Algorithm2
    alg = algorithm.Algorithm2(
        env=env,
        time_horizon=T,
        hyperparameters=hyperparameters,
        is_random_exploration=is_random_exploration,
        explore_limit=explore_limit
    )
    alg.run()

    # After the algorithm has run, calculate regrets
    quality_regret, cost_regret = env.calculate_regret()

    # Compute L2 regularization penalty
    hyperparam_vector = np.array(hyperparameters)
    explore_limit_value = explore_limit
    is_random_exploration_value = 1.0 if is_random_exploration else 0.0
    hyperparam_vector = np.append(hyperparam_vector, [explore_limit_value, is_random_exploration_value])
    penalty = lambda_reg * np.sum(hyperparam_vector ** 2)

    # Compute regularized loss
    regularized_loss = quality_regret + cost_regret + penalty

    # Load best hyperparameters if they exist
    best_loss = None
    if os.path.exists(save_file):
        with open(save_file, 'rb') as f:
            saved_data = pickle.load(f)
            best_loss = saved_data['loss']
    else:
        best_loss = float('inf')

    # Output results
    print(f"\nAlgorithm2 ran with the specified hyperparameters.")
    print("\nSequence of arms pulled and rewards obtained:")
    for t, arm in enumerate(alg.get_arms_history(), 1):
        print(f"Time {t}: Pulled arm {arm}, Reward: {alg.get_rewards_history()[t-1]}")

    print(f"\nEstimated means of each arm:")
    for i, mean in enumerate(alg.get_estimated_means()):
        print(f"Arm {i}: Estimated Mean = {mean:.4f}")

    print(f"\nNumber of pulls per arm:")
    for i, pulls in enumerate(alg.get_n_pulls()):
        print(f"Arm {i}: Number of Pulls = {pulls}")

    print(f"\nTotal quality regret: {quality_regret}")
    print(f"Total cost regret: {cost_regret}")
    print(f"L2 regularization penalty: {penalty}")
    print(f"Regularized loss (quality regret + cost regret + penalty): {regularized_loss}")

    # Compare with best loss
    if is_custom_run:
        if regularized_loss < best_loss:
            # Ask the user if they want to update the best hyperparameters
            update_choice = input("Your custom hyperparameters have better performance than the best saved hyperparameters. Do you want to update the best hyperparameters? (yes/no): ").strip().lower()
            if update_choice in ['yes', 'y']:
                # Save the custom hyperparameters as the new best
                with open(save_file, 'wb') as f:
                    pickle.dump({
                        'hyperparameters': hyperparameters,
                        'explore_limit': explore_limit,
                        'is_random_exploration': is_random_exploration,
                        'loss': regularized_loss,
                        'penalty': penalty,
                        'quality_regret': quality_regret,
                        'cost_regret': cost_regret
                    }, f)
                print("Custom hyperparameters have been saved as the new best hyperparameters.")

                # Add custom hyperparameters to query history and initial points
                add_to_query_history(filename, hyperparameters, explore_limit, is_random_exploration, regularized_loss, penalty, quality_regret, cost_regret, feature_names)
            else:
                print("Best hyperparameters not updated.")
        else:
            if best_loss == float('inf'):
                # No best hyperparameters exist
                update_choice = input("No best hyperparameters exist. Do you want to save your custom hyperparameters as the best? (yes/no): ").strip().lower()
                if update_choice in ['yes', 'y']:
                    with open(save_file, 'wb') as f:
                        pickle.dump({
                            'hyperparameters': hyperparameters,
                            'explore_limit': explore_limit,
                            'is_random_exploration': is_random_exploration,
                            'loss': regularized_loss,
                            'penalty': penalty,
                            'quality_regret': quality_regret,
                            'cost_regret': cost_regret
                        }, f)
                    print("Custom hyperparameters have been saved as the new best hyperparameters.")

                    # Add custom hyperparameters to query history and initial points
                    add_to_query_history(filename, hyperparameters, explore_limit, is_random_exploration, regularized_loss, penalty, quality_regret, cost_regret, feature_names)
                else:
                    print("Best hyperparameters not saved.")
            else:
                print("Your custom hyperparameters did not outperform the best saved hyperparameters.")
    else:
        # For non-custom runs (option 1), we can print the best loss for reference
        print(f"Best saved regularized loss: {best_loss}")

    # Plot cost regret over time
    plot_cost_regret(env, alg, '2', T)  # Passing '2' as algo_choice for Algorithm2

def add_to_query_history(filename, hyperparameters, explore_limit, is_random_exploration, regularized_loss, penalty, quality_regret, cost_regret, feature_names):
    # Function to add hyperparameters to query history
    # Extract 'x' from 'file_x' or 'file_x.txt'
    match = re.search(r'file_(\d+)', filename)
    if match:
        x = match.group(1)
    else:
        x = 'unknown'

    history_file = f'query_history_file_{x}.json'

    # Load existing query history if it exists
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            query_history = json.load(f)
    else:
        query_history = []

    # Add the new entry
    query_entry = {
        'hyperparameters': [float(h) for h in hyperparameters],
        'explore_limit': float(explore_limit),
        'is_random_exploration': bool(is_random_exploration),
        'loss': float(regularized_loss),
        'penalty': float(penalty),
        'quality_regret': float(quality_regret),
        'cost_regret': float(cost_regret)
    }
    query_history.append(query_entry)

    # Save the updated query history
    with open(history_file, 'w') as f:
        json.dump(query_history, f)
    print("Custom hyperparameters have been added to the query history.")

def perform_bayesian_hyperparameter_tuning(filename, T, lambda_reg, feature_names, save_file):
    # Import necessary libraries for Bayesian Optimization
    from skopt import gp_minimize
    from skopt.space import Real, Categorical, Integer
    from skopt.utils import use_named_args
    import json
    import time
    import numpy as np
    import re
    import os
    import pickle
    import sys
    import random  # Use standard random module for consistent data types

    # Prompt the user for time allocation
    try:
        total_time = float(input("Enter the total time for hyperparameter tuning (in seconds): "))
        random_search_time = float(input("Enter the time to dedicate to random search (in seconds): "))
    except ValueError:
        print("Invalid time input. Exiting.")
        sys.exit()

    bayesian_time = total_time - random_search_time
    if bayesian_time <= 0:
        print("Total time must be greater than random search time. Exiting.")
        sys.exit()

    # Define parameter space for hyperparameters
    param_space = [
        Real(-100, 100, name=feature) for feature in feature_names
    ]
    param_space += [
        Real(0.0, 1.0, name='explore_limit'),
        Categorical([True, False], name='is_random_exploration')
    ]

    # File to store black-box query history
    match = re.search(r'file_(\d+)', filename)
    if match:
        file_x = match.group(1)
    else:
        file_x = 'unknown'

    history_file = f'query_history_file_{file_x}.json'
    query_history = []

    # Load existing query history if it exists
    x0 = []
    y0 = []
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            query_history = json.load(f)
            for entry in query_history:
                hyperparameters = entry['hyperparameters']
                explore_limit = entry['explore_limit']
                is_random_exploration = entry['is_random_exploration']
                loss = entry['loss']
                # Prepare initial points
                x0.append(hyperparameters + [explore_limit, is_random_exploration])
                y0.append(loss)
        print(f"Loaded {len(query_history)} previous queries from {history_file}.")

    # Initialize variables for tracking best hyperparameters
    best_loss = float('inf')
    best_hyperparameters = None
    best_quality_regret = None
    best_cost_regret = None
    best_penalty = None
    best_explore_limit = None
    best_is_random_exploration = None

    # Load best hyperparameters if they exist
    if os.path.exists(save_file):
        with open(save_file, 'rb') as f:
            saved_data = pickle.load(f)
            best_loss = saved_data['loss']
            best_hyperparameters = saved_data['hyperparameters']
            best_explore_limit = saved_data['explore_limit']
            best_is_random_exploration = saved_data['is_random_exploration']
            best_penalty = saved_data['penalty']
            best_quality_regret = saved_data['quality_regret']
            best_cost_regret = saved_data['cost_regret']
    else:
        if y0:
            # Set best_loss to the minimum loss in y0
            best_loss = min(y0)
            # Find the index of the minimum loss
            min_index = y0.index(best_loss)
            # Get the corresponding hyperparameters
            best_params = x0[min_index]
            # Extract hyperparameters, explore_limit, and is_random_exploration
            best_hyperparameters = best_params[:-2]
            best_explore_limit = best_params[-2]
            best_is_random_exploration = best_params[-1]
            # Optionally, retrieve penalty, quality_regret, and cost_regret from query_history
            best_entry = query_history[min_index]
            best_penalty = best_entry.get('penalty')
            best_quality_regret = best_entry.get('quality_regret')
            best_cost_regret = best_entry.get('cost_regret')

    # Random Search Phase
    print("\nStarting Random Search Phase...")
    random_start_time = time.time()
    elapsed_time = 0
    run_count = 0

    try:
        while elapsed_time < random_search_time:
            run_count += 1

            # Generate random hyperparameters
            hyperparameters = np.random.uniform(-100, 100, len(feature_names))
            explore_limit = random.uniform(0.0, 1.0)  # Use standard random module
            is_random_exploration = random.choice([True, False])  # Standard Python bool

            # Compute the penalty
            hyperparam_vector = np.array(hyperparameters)
            explore_limit_value = explore_limit
            is_random_exploration_value = 1.0 if is_random_exploration else 0.0
            hyperparam_vector_full = np.append(hyperparam_vector, [explore_limit_value, is_random_exploration_value])
            penalty = lambda_reg * np.sum(hyperparam_vector_full ** 2)

            # Initialize the environment
            env = settings.Environment(filename=filename)

            # Initialize and run Algorithm2
            alg = algorithm.Algorithm2(
                env=env,
                time_horizon=T,
                hyperparameters=hyperparameters,
                is_random_exploration=is_random_exploration,
                explore_limit=explore_limit
            )
            alg.run()

            # After the algorithm has run, calculate regrets
            quality_regret, cost_regret = env.calculate_regret()

            # Compute regularized loss
            regularized_loss = quality_regret + cost_regret + penalty

            # Update best hyperparameters if necessary
            if regularized_loss < best_loss:
                best_loss = regularized_loss
                best_penalty = penalty
                best_hyperparameters = hyperparameters
                best_explore_limit = explore_limit
                best_is_random_exploration = is_random_exploration
                best_quality_regret = quality_regret
                best_cost_regret = cost_regret

                # Save the best hyperparameters to a file
                with open(save_file, 'wb') as f:
                    pickle.dump({
                        'hyperparameters': best_hyperparameters,
                        'explore_limit': best_explore_limit,
                        'is_random_exploration': best_is_random_exploration,
                        'loss': best_loss,
                        'penalty': best_penalty,
                        'quality_regret': best_quality_regret,
                        'cost_regret': best_cost_regret
                    }, f)
                print(f"New best hyperparameters found at random search run {run_count} with regularized loss {best_loss:.4f}")

            # Save query history
            query_entry = {
                'hyperparameters': [float(h) for h in hyperparameters],
                'explore_limit': float(explore_limit),
                'is_random_exploration': bool(is_random_exploration),
                'loss': float(regularized_loss),
                'penalty': float(penalty),
                'quality_regret': float(quality_regret),
                'cost_regret': float(cost_regret)
            }
            query_history.append(query_entry)

            # Add to initial points for Bayesian Optimization
            x0.append(hyperparameters.tolist() + [explore_limit, is_random_exploration])
            y0.append(regularized_loss)

            # Update elapsed time
            elapsed_time = time.time() - random_start_time

            # Print progress every few runs
            if run_count % 1 == 0:
                print(f"Random Search Run {run_count}: Elapsed time {elapsed_time:.2f}s")

    except KeyboardInterrupt:
        print("\nRandom search interrupted by user.")

    # Save query history to file after random search
    with open(history_file, 'w') as f:
        json.dump(query_history, f)
    print(f"Query history updated and saved to {history_file}")

    # Bayesian Optimization Phase
    print("\nStarting Bayesian Optimization Phase...")
    bayesian_start_time = time.time()
    elapsed_time = 0

    # Prepare the objective function for Bayesian Optimization
    @use_named_args(param_space)
    def objective_function(**params):
        start_time = time.time()

        # Extract parameters
        hyperparameters = [params[feature] for feature in feature_names]
        explore_limit = params['explore_limit']
        is_random_exploration = params['is_random_exploration']

        # Compute the penalty
        hyperparam_vector = np.array(hyperparameters)
        explore_limit_value = explore_limit
        is_random_exploration_value = 1.0 if is_random_exploration else 0.0
        hyperparam_vector_full = np.append(hyperparam_vector, [explore_limit_value, is_random_exploration_value])
        penalty = lambda_reg * np.sum(hyperparam_vector_full ** 2)

        # Initialize the environment
        env = settings.Environment(filename=filename)

        # Initialize and run Algorithm2
        alg = algorithm.Algorithm2(
            env=env,
            time_horizon=T,
            hyperparameters=hyperparameters,
            is_random_exploration=is_random_exploration,
            explore_limit=explore_limit
        )
        alg.run()

        # After the algorithm has run, calculate regrets
        quality_regret, cost_regret = env.calculate_regret()

        # Compute regularized loss
        regularized_loss = quality_regret + cost_regret + penalty

        # Save query history
        query_entry = {
            'hyperparameters': [float(h) for h in hyperparameters],
            'explore_limit': float(explore_limit),
            'is_random_exploration': bool(is_random_exploration),
            'loss': float(regularized_loss),
            'penalty': float(penalty),
            'quality_regret': float(quality_regret),
            'cost_regret': float(cost_regret)
        }
        query_history.append(query_entry)

        # Add to initial points for Bayesian Optimization
        x0.append(hyperparameters + [explore_limit, is_random_exploration])
        y0.append(regularized_loss)

        # Update best hyperparameters if necessary
        nonlocal best_loss, best_hyperparameters, best_explore_limit, best_is_random_exploration
        nonlocal best_penalty, best_quality_regret, best_cost_regret
        if regularized_loss < best_loss:
            best_loss = regularized_loss
            best_penalty = penalty
            best_hyperparameters = hyperparameters
            best_explore_limit = explore_limit
            best_is_random_exploration = is_random_exploration
            best_quality_regret = quality_regret
            best_cost_regret = cost_regret

            # Save the best hyperparameters to a file
            with open(save_file, 'wb') as f:
                pickle.dump({
                    'hyperparameters': best_hyperparameters,
                    'explore_limit': best_explore_limit,
                    'is_random_exploration': best_is_random_exploration,
                    'loss': best_loss,
                    'penalty': best_penalty,
                    'quality_regret': best_quality_regret,
                    'cost_regret': best_cost_regret
                }, f)
            print(f"New best hyperparameters found with regularized loss {best_loss:.4f}")

        # Update elapsed time
        nonlocal bayesian_start_time
        elapsed_time = time.time() - bayesian_start_time
        if elapsed_time > bayesian_time:
            raise Exception("Time limit exceeded for Bayesian Optimization.")

        return regularized_loss

    # Run Bayesian Optimization
    try:
        res = gp_minimize(
            func=objective_function,
            dimensions=param_space,
            n_calls=1000,  # Maximum number of function evaluations
            acq_func='EI',
            #acq_func='gp_hedge',
            #acq_func_kwargs={'xi': 0.01},
            x0=x0 if x0 else None,
            y0=y0 if y0 else None,
            n_initial_points=0 if x0 else 5,  # Set to 0 if initial points are provided
            acq_optimizer='auto',
            random_state=42,
            verbose=True
        )
    except Exception as e:
        if str(e) == "Time limit exceeded for Bayesian Optimization.":
            print("Time limit reached for Bayesian Optimization.")
        else:
            print(f"An error occurred during Bayesian Optimization: {e}")

    # Save query history to file after Bayesian Optimization
    with open(history_file, 'w') as f:
        json.dump(query_history, f)
    print(f"Query history updated and saved to {history_file}")

    # After tuning, report the best hyperparameters found
    if best_hyperparameters is not None:
        # Report the formula to calculate score using the best hyperparameters
        formula_terms = [f"{weight:.4f} * {feature}" for weight, feature in zip(best_hyperparameters, feature_names)]
        formula = "Score_i = " + " + ".join(formula_terms)

        # Output results
        print("\nBest hyperparameters found:")
        for weight, feature in zip(best_hyperparameters, feature_names):
            print(f"  Weight for {feature}: {weight:.4f}")
        print(f"Explore limit: {best_explore_limit}")
        print(f"Is random exploration: {best_is_random_exploration}")
        print(f"\nFormula to calculate score:\n{formula}")
        print(f"\nTime horizon T: {T}")
        print(f"Best Regularized loss: {best_loss:.4f}")
        print(f"Best Penalty due to L2 regularization: {best_penalty:.4f}")
        print(f"Best Quality regret: {best_quality_regret:.4f}")
        print(f"Best Cost regret: {best_cost_regret:.4f}")
    else:
        print("No suitable hyperparameters found during tuning.")

def plot_cost_regret(env, alg, algo_choice, T):
    # Extract 'file_x' from the command-line argument
    if len(sys.argv) >= 2:
        filename = sys.argv[1]
        match = re.search(r'file_(\d+)', filename)
        if match:
            file_x = match.group(1)
        else:
            file_x = 'unknown'
    else:
        file_x = 'unknown'

    arms_history = alg.get_arms_history()
    cumulative_cost_regret = []
    cumulative_quality_regret = []

    # Identify feasible arms based on true means
    feasible_arms = [i for i in range(env.get_num_arms()) if env.means[i] >= env.get_threshold()]
    if feasible_arms:
        x = min([env.get_costs()[i] for i in feasible_arms])  # Cost of cheapest feasible arm
    else:
        x = float('inf')  # No feasible arms
        print("No feasible arms with expected reward >= threshold for cost regret calculation.")

    total_quality_regret = 0.0
    total_cost_regret = 0.0

    # Compute step regrets and accumulate
    for t, arm_index in enumerate(arms_history):
        expected_return = env.means[arm_index]  # True mean reward
        step_quality_regret = max(0, env.get_threshold() - expected_return)
        total_quality_regret += step_quality_regret

        cost_i = env.get_costs()[arm_index]
        step_cost_regret = max(0, cost_i - x)
        total_cost_regret += step_cost_regret

        cumulative_quality_regret.append(total_quality_regret)
        cumulative_cost_regret.append(total_cost_regret)

    # Plot settings
    x_values = range(1, len(cumulative_cost_regret) + 1)

    # Plot 1: Cumulative Cost Regret vs Number of Pulls (Regular Scale)
    plt.figure()
    plt.plot(x_values, cumulative_cost_regret)
    plt.xlabel('Number of Pulls')
    plt.ylabel('Cumulative Cost Regret')
    plt.title(f'Algorithm {algo_choice} - Cost Regret vs Number of Pulls\nFile {file_x}')
    plt.grid(True)
    plot_filename = f'Algorithm_{algo_choice}_File_{file_x}_Cost_Regret_Regular.png'
    plt.savefig(plot_filename)
    plt.close()
    print(f"Cumulative cost regret plot (regular scale) saved as {plot_filename}")

    # Plot 2: Cumulative Quality Regret vs Number of Pulls (Regular Scale)
    plt.figure()
    plt.plot(x_values, cumulative_quality_regret)
    plt.xlabel('Number of Pulls')
    plt.ylabel('Cumulative Quality Regret')
    plt.title(f'Algorithm {algo_choice} - Quality Regret vs Number of Pulls\nFile {file_x}')
    plt.grid(True)
    plot_filename = f'Algorithm_{algo_choice}_File_{file_x}_Quality_Regret_Regular.png'
    plt.savefig(plot_filename)
    plt.close()
    print(f"Cumulative quality regret plot (regular scale) saved as {plot_filename}")

    # Plot 3: Cumulative Cost Regret vs Number of Pulls (Log Scale)
    plt.figure()
    plt.plot(x_values, cumulative_cost_regret)
    plt.xlabel('Number of Pulls')
    plt.ylabel('Cumulative Cost Regret')
    plt.xscale('log')
    plt.title(f'Algorithm {algo_choice} - Cost Regret vs Number of Pulls (Log Scale)\nFile {file_x}')
    plt.grid(True)
    plot_filename = f'Algorithm_{algo_choice}_File_{file_x}_Cost_Regret_Log.png'
    plt.savefig(plot_filename)
    plt.close()
    print(f"Cumulative cost regret plot (log scale) saved as {plot_filename}")

    # Plot 4: Cumulative Quality Regret vs Number of Pulls (Log Scale)
    plt.figure()
    plt.plot(x_values, cumulative_quality_regret)
    plt.xlabel('Number of Pulls')
    plt.ylabel('Cumulative Quality Regret')
    plt.xscale('log')
    plt.title(f'Algorithm {algo_choice} - Quality Regret vs Number of Pulls (Log Scale)\nFile {file_x}')
    plt.grid(True)
    plot_filename = f'Algorithm_{algo_choice}_File_{file_x}_Quality_Regret_Log.png'
    plt.savefig(plot_filename)
    plt.close()
    print(f"Cumulative quality regret plot (log scale) saved as {plot_filename}")

if __name__ == "__main__":
    main()
