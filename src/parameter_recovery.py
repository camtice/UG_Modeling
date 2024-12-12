import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import pearsonr
from models import (FehrSchmidtModel, BayesianFehrSchmidtModel, TwoSystemModel, 
                   RWFehrSchmidtModel, PropertyFehrSchmidtModel, 
                   PropertyBayesianFehrSchmidtModel, PropertyRWFehrSchmidtModel)
from fit_parameters import negative_log_likelihood, create_model_instance, fit_participant, stable_softmax
from datetime import datetime

def generate_synthetic_data(model, true_params):
    """Generate synthetic data using a model with known parameters, based on real participant data"""
    # Read the participant data
    data = pd.read_csv('../data/ground/UG_all_role1.csv')
    
    # Get unique participant IDs
    participant_ids = data['ID'].unique()
    
    # Randomly select one participant
    selected_participant = np.random.choice(participant_ids)
    participant_data = data[data['ID'] == selected_participant].copy()
    
    synthetic_data = []
    
    for idx, trial in participant_data.iterrows():
        # Create trial data using the real participant's values
        trial_data = {
            'trial_number': trial['trial_number'],  # Use actual trial number
            'combined_earning': trial['combined_earning'],
            'split_self': trial['split_self'],
            'split_opp': trial['split_opp'],
            'token_self': trial['token_self'],
            'token_opp': trial['token_opp']
        }
        
        # Calculate utility and generate decision
        trial_result = model.calculate_trial_utility(pd.Series(trial_data))
        utility = trial_result['utility']
        
        # Use stable_softmax to calculate p_accept
        p_accept = stable_softmax(utility, model.temperature)
        
        decision = np.random.random() < p_accept
        
        trial_data['accept'] = 1 if decision else 0
        synthetic_data.append(trial_data)
    
    return pd.DataFrame(synthetic_data)

def run_parameter_recovery(model_class, true_params_list, n_iterations):
    """Run parameter recovery analysis"""
    results = []
    
    for iteration in range(n_iterations):
        for true_params in tqdm(true_params_list, desc=f"Iteration {iteration+1}", mininterval=0.01):
            # Create model with true parameters
            true_model = create_model_instance(model_class, true_params)
            
            # Generate synthetic data
            synthetic_data = generate_synthetic_data(true_model, true_params)
            
            # Display the synthetic data for troubleshooting
            # print(f"Synthetic data for true parameters {true_params}:\n", synthetic_data)
            
            # Fit model to recover parameters
            recovered_params = fit_model(synthetic_data, model_class)
            
            # Store results
            result = {
                'iteration': iteration,
                **{f'true_{k}': v for k, v in zip(get_param_names(model_class), true_params)},
                **{f'recovered_{k}': v for k, v in zip(get_param_names(model_class), recovered_params)}
            }
            results.append(result)
    
    return pd.DataFrame(results)

def get_param_names(model_class):
    """Get parameter names for a given model class"""
    if model_class == FehrSchmidtModel:
        return ['alpha', 'temperature']
    elif model_class == BayesianFehrSchmidtModel:
        return ['alpha', 'temperature', 'initial_k']
    elif model_class == TwoSystemModel:
        return ['lambda', 'tau', 'temperature']
    elif model_class == RWFehrSchmidtModel:
        return ['alpha', 'temperature', 'learning_rate']
    # Add property models
    elif model_class == PropertyFehrSchmidtModel:
        return ['alpha', 'temperature']
    elif model_class == PropertyBayesianFehrSchmidtModel:
        return ['alpha', 'temperature', 'initial_k']
    elif model_class == PropertyRWFehrSchmidtModel:
        return ['alpha', 'temperature', 'learning_rate']
    else:
        raise ValueError(f"Unknown model class: {model_class}")

def fit_model(data, model_class):
    """Fit model to data using maximum likelihood estimation"""
    results = fit_participant(data, model_class)
    return [results[param] for param in get_param_names(model_class)]

def plot_recovery_results(results, model_name):
    """Plot parameter recovery results"""
    timestamp = datetime.now().strftime("%m%d%H%M%S")
    
    param_names = [col.replace('true_', '') for col in results.columns 
                  if col.startswith('true_')]
    
    n_params = len(param_names)
    fig, axes = plt.subplots(1, n_params, figsize=(5*n_params, 5))
    if n_params == 1:
        axes = [axes]
    
    for ax, param in zip(axes, param_names):
        true_vals = results[f'true_{param}']
        recovered_vals = results[f'recovered_{param}']
        
        # Calculate correlation
        corr, p_value = pearsonr(true_vals, recovered_vals)
        
        # Add jitter to both true and recovered values
        jitter_amount = (max(true_vals) - min(true_vals)) * 0.01
        true_jittered = true_vals + np.random.normal(0, jitter_amount, len(true_vals))
        recovered_jittered = recovered_vals + np.random.normal(0, jitter_amount, len(recovered_vals))
        
        # Create scatter plot with jittered values
        sns.scatterplot(x=true_jittered, y=recovered_jittered, ax=ax, alpha=0.5, 
                       marker='o', facecolor='none', edgecolor='blue')
        
        # Add diagonal line
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()])
        ]
        ax.plot(lims, lims, 'k--', alpha=0.5)
        
        # Set log scale for temperature parameter
        if param == 'temperature':
            ax.set_xscale('log')
            ax.set_yscale('log')
        
        ax.set_xlabel(f'True {param}')
        ax.set_ylabel(f'Recovered {param}')
        ax.set_title(f'{param}\nr = {corr:.2f}, p = {p_value:.3f}')
    
    plt.suptitle(f'Parameter Recovery Results: {model_name}')
    plt.tight_layout()
    plt.savefig(f'parameter_recovery_{model_name}_{timestamp}.png')
    plt.close()

if __name__ == "__main__":
    # Configuration parameters
    CONFIG = {
        # Parameter space configuration
        'alpha_range': {
            'min': 0.001,
            'max': 20,
            'n_samples': 100
        },
        'temperature_range': {
            'min_exp': -2,  # 10^-2 = 0.01
            'max_exp': 1,   # 10^1 = 10
            'n_samples': 100
        },
        'initial_k_range': {
            'min': 4,
            'max': 4,
            'n_samples': 1
        },
        'learning_rate_range': {
            'min': 0.01,
            'max': 2,
            'n_samples': 100
        },
        'lambda_range': {
            'min': 0,
            'max': 50,
            'n_samples': 100
        },
        'tau_range': {
            'min': 0,
            'max': 20,
            'n_samples': 100
        },
        
        # Recovery configuration
        'n_iterations': 1,
        
        # Models to test
        'models_to_test': ['TwoSystem',]
    }
    
    # Generate parameter space based on configuration
    alphas = np.linspace(CONFIG['alpha_range']['min'], 
                        CONFIG['alpha_range']['max'], 
                        CONFIG['alpha_range']['n_samples'])
    
    # Ensure temperature stays within bounds (0.01 to 10)
    temperatures = np.logspace(CONFIG['temperature_range']['min_exp'],
                             CONFIG['temperature_range']['max_exp'],
                             CONFIG['temperature_range']['n_samples'])
    temperatures = np.clip(temperatures, 0.01, 10)
    
    initial_ks = np.linspace(CONFIG['initial_k_range']['min'],
                            CONFIG['initial_k_range']['max'],
                            CONFIG['initial_k_range']['n_samples'])
    
    # Create parameter combinations for FehrSchmidtModel
    fs_params = []
    for alpha in alphas:
        temp = np.clip(np.random.choice(temperatures), 0.01, 10)
        fs_params.append([alpha, temp])
    
    # Create parameter combinations for BayesianFehrSchmidtModel
    bayesian_fs_params = []
    n_param_combinations = min(100, CONFIG['alpha_range']['n_samples'])
    
    for _ in range(n_param_combinations):
        alpha = np.random.choice(alphas)
        temp = np.clip(np.random.choice(temperatures), 0.01, 10)
        initial_k = np.random.choice(initial_ks)
        bayesian_fs_params.append([alpha, temp, initial_k])
    
    # Create parameter combinations for RWFehrSchmidtModel
    rw_fs_params = []
    n_param_combinations = min(100, CONFIG['alpha_range']['n_samples'])
    
    learning_rates = np.linspace(CONFIG['learning_rate_range']['min'],
                               CONFIG['learning_rate_range']['max'],
                               CONFIG['learning_rate_range']['n_samples'])
    
    for _ in range(n_param_combinations):
        alpha = np.random.choice(alphas)
        temp = np.clip(np.random.choice(temperatures), 0.01, 10)
        learning_rate = np.random.choice(learning_rates)
        rw_fs_params.append([alpha, temp, learning_rate])
    
    # Create parameter combinations for PropertyFehrSchmidtModel (same as FehrSchmidtModel)
    property_fs_params = []
    for alpha in alphas:
        temp = np.clip(np.random.choice(temperatures), 0.01, 10)
        property_fs_params.append([alpha, temp])
    
    # Create parameter combinations for PropertyBayesianFehrSchmidtModel (same as BayesianFehrSchmidtModel)
    property_bayesian_fs_params = []
    n_param_combinations = min(100, CONFIG['alpha_range']['n_samples'])
    
    for _ in range(n_param_combinations):
        alpha = np.random.choice(alphas)
        temp = np.clip(np.random.choice(temperatures), 0.01, 10)
        initial_k = np.random.choice(initial_ks)
        property_bayesian_fs_params.append([alpha, temp, initial_k])
    
    # Create parameter combinations for PropertyRWFehrSchmidtModel (same as RWFehrSchmidtModel)
    property_rw_fs_params = []
    n_param_combinations = min(100, CONFIG['alpha_range']['n_samples'])
    
    for _ in range(n_param_combinations):
        alpha = np.random.choice(alphas)
        temp = np.clip(np.random.choice(temperatures), 0.01, 10)
        learning_rate = np.random.choice(learning_rates)
        property_rw_fs_params.append([alpha, temp, learning_rate])
    
    # Create parameter combinations for TwoSystemModel
    two_system_params = []
    n_param_combinations = min(100, CONFIG['lambda_range']['n_samples'])
    
    lambdas = np.linspace(CONFIG['lambda_range']['min'],
                         CONFIG['lambda_range']['max'],
                         CONFIG['lambda_range']['n_samples'])
    taus = np.linspace(CONFIG['tau_range']['min'],
                        CONFIG['tau_range']['max'],
                        CONFIG['tau_range']['n_samples'])
    
    for _ in range(n_param_combinations):
        lambda_param = np.random.choice(lambdas)
        tau = np.random.choice(taus)
        temp = np.clip(np.random.choice(temperatures), 0.01, 10)
        two_system_params.append([lambda_param, tau, temp])
    
    # Model mapping
    MODEL_MAPPING = {
        'FehrSchmidt': (FehrSchmidtModel, fs_params),
        'TwoSystem': (TwoSystemModel, two_system_params),
        'BayesianFehrSchmidt': (BayesianFehrSchmidtModel, bayesian_fs_params),
        'RWFehrSchmidt': (RWFehrSchmidtModel, rw_fs_params),
        'PropertyFehrSchmidt': (PropertyFehrSchmidtModel, property_fs_params),
        'PropertyBayesianFehrSchmidt': (PropertyBayesianFehrSchmidtModel, property_bayesian_fs_params),
        'PropertyRWFehrSchmidt': (PropertyRWFehrSchmidtModel, property_rw_fs_params),
    }
    
    # Run recovery for configured models
    for model_name in CONFIG['models_to_test']:
        if model_name not in MODEL_MAPPING:
            print(f"Warning: {model_name} not found in MODEL_MAPPING. Skipping...")
            continue
            
        model_class, param_sets = MODEL_MAPPING[model_name]
        print(f"\nRunning parameter recovery for {model_name}")
        results = run_parameter_recovery(model_class, param_sets, CONFIG['n_iterations'])
        
        # Save results
        results.to_csv(f'parameter_recovery_{model_name}.csv', index=False)
        
        # Plot results
        plot_recovery_results(results, model_name)