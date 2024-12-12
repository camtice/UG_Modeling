import numpy as np
from scipy.optimize import minimize, basinhopping
from models import (FehrSchmidtModel, TwoSystemModel, BayesianFehrSchmidtModel, 
                   RWFehrSchmidtModel, PropertyFehrSchmidtModel, 
                   PropertyBayesianFehrSchmidtModel, PropertyRWFehrSchmidtModel)
import pandas as pd
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

################## CHANGE TO ACTUAL SOFTMAX AND NOT GENERAL LOGISTIC ############################
def stable_softmax(utility, temperature):
    """Compute a stable softmax function to avoid overflow."""
    # Add a small epsilon to temperature to avoid division by zero
    epsilon = 1e-10
    temperature = max(temperature, epsilon)
    
    # Clip the utility values to avoid overflow
    utility_clipped = np.clip(utility / temperature, -100, 100)  # Adjust the clipping range as needed
    return 1 / (1 + np.exp(-utility_clipped))

def softmax_probability(utility, temperature):
    """Calculate the probability of accepting using stable softmax function"""
    return stable_softmax(utility, temperature)

def negative_log_likelihood(params, data, model_class):
    """
    Calculate negative log likelihood for model predictions
    
    Parameters:
    params: list/array of parameters (varies by model)
    data: DataFrame containing participant's decisions
    model_class: Class of the utility model to use
    """
    # Get model-specific parameter mapping
    model = create_model_instance(model_class, params)
    
    total_ll = 0
    for _, trial in data.iterrows():
        trial_results = model.calculate_trial_utility(trial)
        utility = trial_results['utility']
        
        # Calculate probability of accepting
        p_accept = softmax_probability(utility, model.temperature)
        
        actual_decision = trial['accept']
        
        # Add to log likelihood
        p = p_accept if actual_decision == 1 else (1 - p_accept)
        total_ll += np.log(p + 1e-10)
    
    return -total_ll

def create_model_instance(model_class, params):
    """Create model instance based on model class and parameters"""
    if model_class == RWFehrSchmidtModel:
        alpha, temperature, learning_rate = params
        return model_class(alpha=alpha, beta=0, temperature=temperature, learning_rate=learning_rate)
    elif model_class == FehrSchmidtModel:
        alpha, temperature = params
        return model_class(alpha=alpha, beta=0, temperature=temperature)
    elif model_class == TwoSystemModel:
        lambda_param, tau, temperature = params
        return model_class(lambda_param=lambda_param, tau=tau, temperature=temperature)
    elif model_class == BayesianFehrSchmidtModel:
        alpha, temperature, initial_k = params
        return model_class(alpha=alpha, beta=0, temperature=temperature, initial_k=initial_k)
    elif model_class == PropertyFehrSchmidtModel:
        alpha, temperature = params
        return model_class(alpha=alpha, beta=0, temperature=temperature)
    elif model_class == PropertyBayesianFehrSchmidtModel:
        alpha, temperature, initial_k = params
        return model_class(alpha=alpha, beta=0, temperature=temperature, initial_k=initial_k)
    elif model_class == PropertyRWFehrSchmidtModel:
        alpha, temperature, learning_rate = params
        return model_class(alpha=alpha, beta=0, temperature=temperature, learning_rate=learning_rate)
    else:
        raise ValueError(f"Unsupported model class: {model_class}")

def get_model_config(model_class):
    """Get model-specific configuration for parameter fitting"""
    if model_class == RWFehrSchmidtModel:
        return {
            'bounds': [
                (0, 20),    # alpha
                (0.01, 10), # temperature
                (0.01, 2)   # learning_rate
            ],
            'param_names': ['alpha', 'temperature', 'learning_rate']
        }
    elif model_class == FehrSchmidtModel:
        return {
            'bounds': [
                (0.001, 5),  # alpha
                (0.01, 10)   # temperature
            ],
            'param_names': ['alpha', 'temperature']
        }
    elif model_class == TwoSystemModel:
        return {
            'bounds': [
                (0, 50),      # lambda
                (0, 20),      # tau
                (0.001, 10)    # temperature
            ],
            'param_names': ['lambda', 'tau', 'temperature']
        }
    elif model_class == BayesianFehrSchmidtModel:
        return {
            'bounds': [
                (0.001, 20),  # alpha
                (0.01, 10),   # temperature
                (4, 4)       # initial_k
            ],
            'param_names': ['alpha', 'temperature', 'initial_k']
        }
    elif model_class == PropertyFehrSchmidtModel:
        return {
            'bounds': [
                (0.001, 5),  # alpha
                (0.01, 10)   # temperature
            ],
            'param_names': ['alpha', 'temperature']
        }
    elif model_class == PropertyBayesianFehrSchmidtModel:
        return {
            'bounds': [
                (0.001, 20),  # alpha
                (0.01, 10),   # temperature
                (1, 1000)       # initial_k
            ],
            'param_names': ['alpha', 'temperature', 'initial_k']
        }
    elif model_class == PropertyRWFehrSchmidtModel:
        return {
            'bounds': [
                (0, 30),    # alpha
                (0.01, 10), # temperature
                (0.01, 2)   # learning_rate
            ],
            'param_names': ['alpha', 'temperature', 'learning_rate']
        }
    else:
        raise ValueError(f"Unsupported model class: {model_class}")

def random_starting_points(bounds, n_starts=10):
    """Generate random starting points within bounds"""
    starts = []
    for _ in range(n_starts):
        point = []
        for (low, high) in bounds:
            # Use log-uniform sampling for temperature parameters
            if 'temperature' in bounds and high < 100:
                point.append(np.exp(np.random.uniform(np.log(low), np.log(high))))
            else:
                point.append(np.random.uniform(low, high))
        starts.append(point)
    return starts

def fit_participant(participant_data, model_class, fixed_temperature=None):
    """
    Fit model parameters for a single participant using multiple random starts and basin-hopping
    
    Parameters:
    participant_data: DataFrame containing participant's decisions
    model_class: Class of the utility model to use
    fixed_temperature: If provided, fixes the temperature parameter to this value
    """
    config = get_model_config(model_class)
    
    # If temperature is fixed, remove it from bounds and param names
    if fixed_temperature is not None:
        bounds = [b for b, name in zip(config['bounds'], config['param_names']) 
                 if name != 'temperature']
        param_names = [name for name in config['param_names'] if name != 'temperature']
    else:
        bounds = config['bounds']
        param_names = config['param_names']
    
    # Generate multiple random starting points
    initial_points = random_starting_points(bounds, n_starts=3)
    
    best_result = None
    best_likelihood = float('-inf')
    
    # Modify the objective function to handle fixed temperature
    def modified_objective(params):
        if fixed_temperature is not None:
            # Insert fixed temperature back into params at the correct position
            full_params = []
            temp_idx = config['param_names'].index('temperature')
            for i, name in enumerate(config['param_names']):
                if name == 'temperature':
                    full_params.append(fixed_temperature)
                else:
                    full_params.append(params[len(full_params)])
        else:
            full_params = params
        return negative_log_likelihood(full_params, participant_data, model_class)
    
    # Basin-hopping and minimizer settings remain the same
    # basin_kwargs = {
    #     "T": 1.0,
    #     "stepsize": 0.5,
    #     "niter": 10,
    #     "interval": 5,
    # }
    
    minimizer_kwargs = {
        "method": "L-BFGS-B",
        "bounds": bounds,
        "options": {
            "maxiter": 1000,
            "ftol": 1e-6,
        }
    }
    
    # Try optimization from each starting point
    for init_params in initial_points:
        try:
            # Commenting out basin hopping
            # result = basinhopping(
            #     modified_objective,
            #     init_params,
            #     minimizer_kwargs=minimizer_kwargs,
            #     **basin_kwargs
            # )
            
            # Use only the minimizer
            result = minimize(
                modified_objective,
                init_params,
                **minimizer_kwargs
            )
            
            if result.success and -result.fun > best_likelihood:
                best_result = result
                best_likelihood = -result.fun
                
        except Exception as e:
            logging.warning(f"Optimization attempt failed: {str(e)}")
            continue
    
    if best_result is None:
        logging.warning(f"All optimizations failed for participant {participant_data['ID'].iloc[0]}")
        return None
    
    # Create results dictionary
    fit_results = {}
    param_idx = 0
    for name in config['param_names']:
        if name == 'temperature' and fixed_temperature is not None:
            fit_results[name] = fixed_temperature
        else:
            fit_results[name] = best_result.x[param_idx]
            param_idx += 1
            
    fit_results.update({
        'success': best_result.success,
        'log_likelihood': -best_result.fun
    })
    
    return fit_results

def fit_all_participants(data, model_class):
    """Fit model parameters for all participants"""
    results = []
    
    # Get unique participants first
    unique_participants = data['ID'].unique()
    
    # Create progress bar
    for participant_id in tqdm(unique_participants, desc="Fitting participants"):
        participant_data = data[data['ID'] == participant_id]
        
        # Fit parameters for this participant
        fit_results = fit_participant(participant_data, model_class)
        
        # Store results
        results.append({
            'participant_id': participant_id,
            **fit_results
        })
    
    return pd.DataFrame(results)

def plot_parameter_distributions(results_df, model_class):
    """Plot the distribution of model parameters."""
    param_names = get_model_config(model_class)['param_names']
    n_params = len(param_names)
    
    # Create subplots for all parameters
    fig, axs = plt.subplots(1, n_params, figsize=(8*n_params, 6))  # Adjust figure size based on number of parameters
    
    # If only one parameter, axs won't be an array, so convert it
    if n_params == 1:
        axs = [axs]
    
    for ax, param in zip(axs, param_names):
        ax.hist(results_df[param], bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax.set_title(f'Distribution of {param}')
        ax.set_xlabel(param)
        ax.set_ylabel('Frequency')
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Load your data
    raw_data = pd.read_csv('../data/ground/UG_all_role1.csv')
    
    #raw_data = pd.read_csv('../data/ground/UG_raw_data_with_trials.csv')
    
    # Select model to fit
    model_to_fit = TwoSystemModel  # Change this to switch models
    
    # Fit models for all participants
    results_df = fit_all_participants(raw_data, model_to_fit)
    
    # Save results with model name in filename
    model_name = model_to_fit.__name__.lower()
    results_df.to_csv(f'{model_name}_fits.csv', index=False)
    
    # Print summary statistics
    print(f"\nFitting Results Summary for {model_to_fit.__name__}:")
    print(results_df.describe())
    
    # Plot parameter distributions
    plot_parameter_distributions(results_df, model_to_fit)