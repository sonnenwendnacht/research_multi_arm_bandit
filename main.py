import sys
import numpy as np
import settings
import algorithm  # Ensure this imports both Algorithm1 and Algorithm2
import time
import os
import pickle  # For saving and loading best hyperparameters
import re

def main():
    # Initialize variables
    T = 1000  # Time horizon
    param_range = (-100, 100)
    # Regularization parameter
    lambda_reg = 0.0001  # Adjust as needed 

    num_runs = 50  # Reduced number of runs per hyperparameter set to speed up iterations

    feature_names = [
        "reward_threshold",
        "sample_mean",
        "sample_variance",
        "sample_median",
        "sample_max",
        "sample_min",
        "cost_i",

        "missing_reward",
        "missing_reward_percentage",
        "missing_reward_percentage_log",
        "missing_reward_percentage_exp",
        "reward_overflow",
        "reward_overflow_percentage",
        "reward_overflow_percentage_log",
        "reward_overflow_percentage_exp",

        "pull_number",
        "ucb",

        "remaining_pulls",
        "remaining_percentage",

        "other_cost1",
        "exist1",
        "sample_mean1",
        "missing_reward1",
        "missing_reward_percentage1",
        "missing_reward_percentage_log1",
        "missing_reward_percentage_exp1",
        "reward_overflow1",
        "reward_overflow_percentage1",
        "reward_overflow_percentage_log1",
        "reward_overflow_percentage_exp1",
        "pull_number1",
        "ucb1",
        "remaining_percentage1",

        "other_cost2",
        "exist2",
        "sample_mean2",
        "missing_reward2",
        "missing_reward_percentage2",
        "missing_reward_percentage_log2",
        "missing_reward_percentage_exp2",
        "reward_overflow2",
        "reward_overflow_percentage2",
        "reward_overflow_percentage_log2",
        "reward_overflow_percentage_exp2",
        "pull_number2",
        "ucb2",
        "remaining_percentage2",

        "zero"
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
    save_file = f'best_hyperparameters_{x}.pkl'  # Added .pkl extension for consistency

    # Prompt the user to select the algorithm or delete saved data
    algo_choice = input("Enter which algorithm to run (1 or 2), or type 'DELETE' to remove saved hyperparameters: ").strip()

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

    elif algo_choice not in ['1', '2']:
        print("Invalid choice. Exiting.")
        sys.exit()

    if algo_choice == '1':
        # Initialize the environment
        env = settings.Environment(filename=filename)

        # Initialize and run Algorithm1
        alg = algorithm.Algorithm1(env=env, time_horizon=T)
        alg.run()

        # After the algorithm has run, calculate regrets
        quality_regret, cost_regret = env.calculate_regret()

        # Output results
        print(f"\nAlgorithm selected arm: {alg.get_chosen_arm()}")
        print(f"Estimated means: {alg.get_estimated_means()}")
        print(f"Number of pulls per arm: {alg.get_n_pulls()}")
        print(f"Total quality regret: {quality_regret}")
        print(f"Total cost regret: {cost_regret}")

    elif algo_choice == '2':
        # Ask the user how long they want the program to run (in seconds)
        try:
            time_limit = float(input("Enter the time limit for hyperparameter tuning (in seconds): "))
        except ValueError:
            print("Invalid time limit. Exiting.")
            sys.exit()

        best_loss = float('inf')
        best_hyperparameters = None
        best_quality_regret = None
        best_cost_regret = None
        best_penalty = None
        best_explore_limit = None
        best_is_random_exploration = None

        # Check if there's a saved best hyperparameters file
        if os.path.exists(save_file):
            with open(save_file, 'rb') as f:
                saved_data = pickle.load(f)
                best_hyperparameters = saved_data['hyperparameters']
                best_explore_limit = saved_data['explore_limit']
                best_is_random_exploration = saved_data['is_random_exploration']

            print("Recomputing the loss for the loaded hyperparameters to ensure fairness...")

            # Recompute the loss using the loaded hyperparameters
            total_loss = 0
            total_quality_regret = 0
            total_cost_regret = 0

            # Compute the penalty
            hyperparam_vector = np.array(best_hyperparameters)
            explore_limit_value = best_explore_limit
            is_random_exploration_value = 1.0 if best_is_random_exploration else 0.0
            hyperparam_vector = np.append(hyperparam_vector, [explore_limit_value, is_random_exploration_value])
            penalty = lambda_reg * np.sum(hyperparam_vector ** 2)

            for _ in range(num_runs):
                # Initialize the environment for each run
                env = settings.Environment(filename=filename)

                # Initialize and run Algorithm2
                alg = algorithm.Algorithm2(
                    env=env,
                    time_horizon=T,
                    hyperparameters=best_hyperparameters,
                    is_random_exploration=best_is_random_exploration,
                    explore_limit=best_explore_limit
                )
                alg.run()

                # After the algorithm has run, calculate regrets
                quality_regret, cost_regret = env.calculate_regret()
                total_loss += (quality_regret + cost_regret)
                total_quality_regret += quality_regret
                total_cost_regret += cost_regret

            # Compute average loss and regrets for these hyperparameters
            avg_loss = total_loss / num_runs
            avg_quality_regret = total_quality_regret / num_runs
            avg_cost_regret = total_cost_regret / num_runs

            # Add penalty to the average loss
            regularized_loss = avg_loss + penalty

            best_loss = regularized_loss  # Update best_loss
            best_penalty = penalty
            best_quality_regret = avg_quality_regret
            best_cost_regret = avg_cost_regret

            print(f"Loaded hyperparameters have average regularized loss: {best_loss:.4f}")
            print(f"Penalty due to L2 regularization: {best_penalty:.4f}")
            print(f"Average quality regret: {best_quality_regret:.4f}")
            print(f"Average cost regret: {best_cost_regret:.4f}")
        else:
            print("No previous best hyperparameters found. Starting fresh.")

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
                if run_count % 100 == 0:
                    print(f"Run {run_count}: Elapsed time {elapsed_time:.2f}s")

        except KeyboardInterrupt:
            print("\nHyperparameter tuning interrupted by user.")

        # Check if best_hyperparameters is not None before proceeding
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
