import numpy as np
import pandas as pd
from models import FehrSchmidtModel, TwoSystemModel
from tqdm import tqdm
from itertools import product

def stable_softmax(utility, temperature):
    """Compute a stable softmax function to avoid overflow."""
    # Clip the utility values to avoid overflow
    utility_clipped = np.clip(utility / temperature, -500, 500)  # Adjust the clipping range as needed
    return 1 / (1 + np.exp(-utility_clipped))

def negative_log_likelihood(params, data, model_class):
    """
    Calculate negative log likelihood for model predictions
    
    Parameters:
    params: list/array of parameters (varies by model)
    data: DataFrame containing participant's decisions
    model_class: Class of the utility model to use
    """
    model = create_model_instance(model_class, params)
    
    total_ll = 0
    for _, trial in data.iterrows():
        trial_results = model.calculate_trial_utility(trial)
        utility = trial_results['utility']
        p_accept = stable_softmax(utility, model.temperature)
        actual_decision = trial['accept']
        p = p_accept if actual_decision == 1 else (1 - p_accept)
        total_ll += np.log(p + 1e-10)
    
    return -total_ll

def create_model_instance(model_class, params):
    """Create model instance based on model class and parameters"""
    if model_class == FehrSchmidtModel:
        alpha, temperature = params
        return model_class(alpha=alpha, beta=0, temperature=temperature)
    elif model_class == TwoSystemModel:
        lambda_param, tau, temperature = params
        return model_class(lambda_param=lambda_param, tau=tau, temperature=temperature)
    else:
        raise ValueError(f"Unsupported model class: {model_class}")

def get_model_config(model_class):
    """Get model-specific configuration for parameter fitting"""
    if model_class == FehrSchmidtModel:
        return {
            'param_ranges': {
                'alpha': np.linspace(0, 10, 21),       # 0 to 10 in 0.5 steps
                'temperature': np.linspace(0.1, 10, 20) # 0.1 to 10 in ~0.5 steps
            },
            'param_names': ['alpha', 'temperature']
        }
    elif model_class == TwoSystemModel:
        return {
            'param_ranges': {
                'lambda': np.linspace(0, 20, 51),    
                'tau': np.linspace(0, 1, 21),   
                'temperature': [9] 
            },
            'param_names': ['lambda', 'tau', 'temperature']
        }
    else:
        raise ValueError(f"Unsupported model class: {model_class}")

def fit_participant_brute_force(participant_data, model_class):
    """Fit model parameters for a single participant using brute force search"""
    
    config = get_model_config(model_class)
    param_ranges = config['param_ranges']
    param_names = config['param_names']
    
    # Create grid of all parameter combinations
    param_values = [param_ranges[param] for param in param_names]
    param_combinations = list(product(*param_values))
    
    best_ll = float('-inf')
    best_params = None
    
    # Try all parameter combinations
    for params in tqdm(param_combinations, desc="Testing parameters", leave=False):
        ll = -negative_log_likelihood(params, participant_data, model_class)
        if ll > best_ll:
            best_ll = ll
            best_params = params
    
    # Create results dictionary
    fit_results = {
        name: value for name, value in zip(param_names, best_params)
    }
    fit_results.update({
        'success': True,
        'log_likelihood': best_ll
    })
    
    return fit_results

def fit_all_participants(data, model_class):
    """Fit model parameters for all participants"""
    results = []
    unique_participants = data['ID'].unique()
    
    for participant_id in tqdm(unique_participants, desc="Fitting participants"):
        participant_data = data[data['ID'] == participant_id]
        fit_results = fit_participant_brute_force(participant_data, model_class)
        results.append({
            'participant_id': participant_id,
            **fit_results
        })
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    # Load your data
    raw_data = pd.read_csv('data/ground/UG_raw_data_with_trials.csv')
    
    # Select model to fit
    model_to_fit = TwoSystemModel  # Change this to FehrSchmidtModel or TwoSystemModel
    
    # Print parameter ranges being searched
    config = get_model_config(model_to_fit)
    print(f"\nParameter ranges for {model_to_fit.__name__}:")
    for param_name, param_range in config['param_ranges'].items():
        print(f"{param_name}: {param_range[0]:.2f} to {param_range[-1]:.2f} (steps: {len(param_range)})")
    print(f"Total parameter combinations: {np.prod([len(r) for r in config['param_ranges'].values()]):,}\n")
    
    # Fit models for all participants
    results_df = fit_all_participants(raw_data, model_to_fit)
    
    # Load group information and merge with results
    group_info = pd.read_csv('data/ground/UG_raw_data_with_trials.csv')  # Adjust path as needed
    results_df = results_df.merge(group_info[['ID', 'group']], 
                                left_on='participant_id', 
                                right_on='ID', 
                                how='left')
    
    # Save complete results with model name in filename
    model_name = model_to_fit.__name__.lower()
    results_df.to_csv(f'{model_name}_fits_brute_force.csv', index=False)
    
    # Print summary statistics split by group
    print(f"\nFitting Results Summary for {model_to_fit.__name__}:")
    print("\nControl Group Summary:")
    print(results_df[results_df['group'] == 'Control'].describe())
    print("\nCocaine Group Summary:")
    print(results_df[results_df['group'] == 'Cocaine'].describe())
    
    # Print mean parameters for each group
    param_names = config['param_names']
    print("\nMean Parameters by Group:")
    for group in ['Control', 'Cocaine']:
        group_data = results_df[results_df['group'] == group]
        print(f"\n{group} Group (n={len(group_data)}):")
        for param in param_names:
            mean_val = group_data[param].mean()
            std_val = group_data[param].std()
            print(f"{param}: {mean_val:.3f} ± {std_val:.3f}")