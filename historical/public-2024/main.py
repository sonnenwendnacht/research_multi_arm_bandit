# main.py

import sys
import numpy as np
import settings
import algorithm  # Ensure this imports Algorithm1, Algorithm2, and Algorithm3
import time
import os
import pickle  # For saving and loading best hyperparameters
import re

def main():
    # Initialize variables
    T = 50000  # Adjusted time horizon for demonstration purposes
    param_range = (-100, 100)
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
        print("Usage: python3 main.py [filename]")
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
        confirmation = input(f"Are you sure you want to delete the saved best hyperparameters for file_{x}? Type 'CONTINUE DELETE' to proceed: ").strip()
        if confirmation == 'CONTINUE DELETE':
            if os.path.exists(save_file):
                os.remove(save_file)
                print(f"Saved best hyperparameters for file_{x} have been deleted.")
            else:
                print(f"No saved best hyperparameters file found for file_{x}.")
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
                run_algorithm2_and_display_results(env, T, best_hyperparameters, best_explore_limit, best_is_random_exploration, save_file, lambda_reg, feature_names, is_custom_run=False)
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
            run_algorithm2_and_display_results(env, T, hyperparameters, explore_limit, is_random_exploration, save_file, lambda_reg, feature_names, is_custom_run=True)
        elif action_choice == '3':
            # Perform hyperparameter tuning
            # The existing code for hyperparameter tuning can be placed here
            # No modifications needed; minimal output is acceptable during tuning
            perform_hyperparameter_tuning(filename, T, param_range, lambda_reg, feature_names, save_file)
        else:
            print("Invalid choice. Exiting.")
            sys.exit()

def run_algorithm2_and_display_results(env, T, hyperparameters, explore_limit, is_random_exploration, save_file, lambda_reg, feature_names, is_custom_run):
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
                else:
                    print("Best hyperparameters not saved.")
            else:
                print("Your custom hyperparameters did not outperform the best saved hyperparameters.")
    else:
        # For non-custom runs (option 1), we can print the best loss for reference
        print(f"Best saved regularized loss: {best_loss}")

def perform_hyperparameter_tuning(filename, T, param_range, lambda_reg, feature_names, save_file):
    # Existing code for hyperparameter tuning
    num_runs = 10  # Adjust as needed
    # Initialize variables for tracking best hyperparameters
    best_loss = float('inf')
    best_hyperparameters = None
    best_quality_regret = None
    best_cost_regret = None
    best_penalty = None
    best_explore_limit = None
    best_is_random_exploration = None

    # Ask the user how long they want the program to run (in seconds)
    try:
        time_limit = float(input("Enter the time limit for hyperparameter tuning (in seconds): "))
    except ValueError:
        print("Invalid time limit. Exiting.")
        sys.exit()

    start_time = time.time()
    elapsed_time = 0
    run_count = 0

    print("Starting hyperparameter tuning... Press Ctrl+C to interrupt.")

    try:
        while elapsed_time < time_limit:
            run_count += 1

            # Generate a random set of hyperparameters
            hyperparameters = np.random.uniform(param_range[0], param_range[1], len(feature_names))

            # Randomly choose explore_limit in a reasonable range
            explore_limit = np.random.uniform(0.0, 1.0)  # Adjust range as needed

            # Randomly choose is_random_exploration
            is_random_exploration = np.random.choice([True, False])

            total_loss = 0
            total_quality_regret = 0
            total_cost_regret = 0

            # Compute the penalty
            hyperparam_vector = np.array(hyperparameters)
            explore_limit_value = explore_limit
            is_random_exploration_value = 1.0 if is_random_exploration else 0.0
            hyperparam_vector = np.append(hyperparam_vector, [explore_limit_value, is_random_exploration_value])
            penalty = lambda_reg * np.sum(hyperparam_vector ** 2)

            # Run Algorithm2 multiple times with these hyperparameters
            for _ in range(num_runs):
                # Initialize the environment for each run
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
                total_loss += (quality_regret + cost_regret)
                total_quality_regret += quality_regret
                total_cost_regret += cost_regret

            # Compute average loss and regrets for this set of hyperparameters
            avg_loss = total_loss / num_runs
            avg_quality_regret = total_quality_regret / num_runs
            avg_cost_regret = total_cost_regret / num_runs

            # Add penalty to the average loss
            regularized_loss = avg_loss + penalty

            # Check if this is the best hyperparameters
            if regularized_loss < best_loss:
                best_loss = regularized_loss
                best_penalty = penalty
                best_hyperparameters = hyperparameters
                best_explore_limit = explore_limit
                best_is_random_exploration = is_random_exploration
                best_quality_regret = avg_quality_regret
                best_cost_regret = avg_cost_regret
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
                print(f"New best hyperparameters found at run {run_count} with regularized loss {best_loss:.4f}")
                print(f"Penalty due to L2 regularization: {best_penalty:.4f}")
                print(f"Average quality regret: {best_quality_regret:.4f}")
                print(f"Average cost regret: {best_cost_regret:.4f}")

            # Update elapsed time
            elapsed_time = time.time() - start_time

            # Print progress every few runs
            if run_count % 1 == 0:
                print(f"Run {run_count}: Elapsed time {elapsed_time:.2f}s")

    except KeyboardInterrupt:
        print("\nHyperparameter tuning interrupted by user.")

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
        print(f"Best Average regularized loss: {best_loss:.4f}")
        print(f"Best Penalty due to L2 regularization: {best_penalty:.4f}")
        print(f"Best Average quality regret: {best_quality_regret:.4f}")
        print(f"Best Average cost regret: {best_cost_regret:.4f}")
        print(f"Total hyperparameter sets evaluated: {run_count}")
    else:
        print("No suitable hyperparameters found during tuning.")

if __name__ == "__main__":
    main()
