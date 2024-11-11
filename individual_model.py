import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class UtilityModel:
    """Base class for utility models"""
    def __init__(self, alpha, beta, temperature):
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature

    def calculate_utility(self, own_amount, other_amount):
        raise NotImplementedError("Subclasses must implement calculate_utility")
    
    def calculate_trial_utility(self, row):
        raise NotImplementedError("Subclasses must implement calculate_trial_utility")


class FehrSchmidtModel(UtilityModel):
    def calculate_utility(self, own_amount, other_amount):
        envy = max(other_amount - own_amount, 0)
        guilt = max(own_amount - other_amount, 0)
        return own_amount - (self.alpha * envy) - (self.beta * guilt)
    
    def calculate_trial_utility(self, row):
        utility = self.calculate_utility(row['split_self'], row['split_opp'])
        return {
            'utility': utility,
            'comparison_offer': row['split_self'],
            'additional_metrics': {}
        }


class BayesianFehrSchmidtModel(UtilityModel):
    def __init__(self, alpha, beta, temperature, initial_k=1, initial_v=1, initial_sigma2=0.04):
        super().__init__(alpha, beta, temperature)
        self.mu_hat = 0.5  # Initial expected proportion
        self.k = initial_k
        self.v = initial_v
        self.sigma2_hat = initial_sigma2

    def calculate_utility(self, own_amount, expected_offer):
        envy = max(expected_offer - own_amount, 0)
        guilt = max(own_amount - expected_offer, 0)
        return own_amount - (self.alpha * envy) - (self.beta * guilt)
    
    def calculate_trial_utility(self, row):
        total_pot = row['combined_earning']
        current_mu = self.mu_hat  # Store current mu_hat before update
        expected_offer = current_mu * total_pot
        
        utility = self.calculate_utility(row['split_self'], expected_offer)
        
        # Calculate observed proportion
        observed_proportion = row['split_self'] / total_pot
        
        # Store previous values for debugging
        prev_k = self.k
        prev_v = self.v
        prev_sigma2 = self.sigma2_hat
        
        # Perform Bayesian update
        self.mu_hat, self.k, self.v, self.sigma2_hat = bayesian_update(
            observed_proportion,
            current_mu,
            prev_k,
            prev_v,
            prev_sigma2
        )
        
        return {
            'utility': utility,
            'comparison_offer': expected_offer,
            'additional_metrics': {
                'Expected Proportion': round(current_mu, 2),  # Use current_mu (before update)
                'Variance': round(self.sigma2_hat, 4),
                'Observed Proportion': round(observed_proportion, 2),
                'Prior mu': round(current_mu, 2),  # Use same current_mu
                'Prior k': round(prev_k, 2),
                'Prior v': round(prev_v, 2),
                'Prior sigma2': round(prev_sigma2, 4),
                'Delta mu': round(self.mu_hat - current_mu, 4),
                'Update Weight': round(1/self.k, 4),
                'Expected Offer': round(expected_offer, 2),
                'Total Pot': total_pot
            }
        }


def bayesian_update(x_t_proportion, mu_hat_prev, k_prev, v_prev, sigma2_hat_prev):
    """
    Perform Bayesian update for the mean and variance of the offer proportions.

    Parameters:
    x_t_proportion: observed offer proportion at trial t (e.g., 0.5 for a 50-50 split)
    mu_hat_prev: previous estimated mean proportion μ̂_{t-1}
    k_prev: previous k_{t-1}
    v_prev: previous v_{t-1}
    sigma2_hat_prev: previous estimated variance σ̂_{t-1}^2

    Returns:
    mu_hat_t: updated estimated mean proportion μ̂_t
    k_t: updated k_t
    v_t: updated v_t
    sigma2_hat_t: updated estimated variance σ̂_t^2
    """
    k_t = k_prev + 1
    v_t = v_prev + 1
    
    # Update mean estimate (weighted average of previous estimate and new observation)
    mu_hat_t = (k_prev / k_t) * mu_hat_prev + (1 / k_t) * x_t_proportion
    
    # Update variance estimate
    variance_update = v_prev * sigma2_hat_prev + (k_prev / k_t) * (x_t_proportion - mu_hat_prev) ** 2
    sigma2_hat_t = variance_update / v_t
    
    return mu_hat_t, k_t, v_t, sigma2_hat_t

def simulate_responder_decision(utility, temperature):
    """
    Simulate responder's decision using softmax decision rule
    
    Parameters:
    utility: calculated utility value
    temperature: softmax temperature parameter (default=1.0)
    
    Returns:
    tuple: (decision (bool), acceptance_probability)
    """
    # Calculate probability of accepting using softmax
    acceptance_prob = 1 / (1 + np.exp(-utility/temperature))
    
    # Make probabilistic decision
    decision = np.random.random() < acceptance_prob
    
    return decision, acceptance_prob

def analyze_individual_decisions(participant_id, utility_model, data, role='responder'):
    """Analyze decisions for a specific participant and compare with model predictions"""
    # Filter for participant and role
    individual_data = data[data['ID'] == participant_id].copy()
    individual_data = individual_data[individual_data['trial_role'] == (1 if role == 'responder' else 2)]
    
    results = []
    for _, row in individual_data.iterrows():
        # Calculate utility using model-specific logic
        trial_result = utility_model.calculate_trial_utility(row)
        utility = trial_result['utility']
        comparison_offer = trial_result['comparison_offer']
        
        # Simulate decision
        decision, acceptance_prob = simulate_responder_decision(utility, utility_model.temperature)
        
        # Build result dictionary
        result_dict = {
            'Trial': row['trial_number'],
            'Trial Type': row['trial_type'],
            'Total Pot': row['combined_earning'],
            'Tokens Self': row['token_self'],
            'Tokens Other': row['token_opp'],
            'Individual Offer': row['split_self'],
            'Opponent Offer': row['split_opp'],
            'Comparison Offer': round(comparison_offer, 2),
            'Actual D': 'Accept' if row['accept'] == 1 else 'Reject',
            'Model D': 'Accept' if decision else 'Reject',
            'Utility': round(utility, 2),
            'Accept Probability': round(acceptance_prob, 2)
        }
        
        # Add any additional model-specific metrics
        result_dict.update(trial_result['additional_metrics'])
        results.append(result_dict)
    
    return pd.DataFrame(results)

def display_analysis_results(results, participant_id, model_name, model_params=None):
    print(f"\nAnalysis for Participant {participant_id}")
    if model_params:
        param_str = ", ".join([f"{k}={v}" for k, v in model_params.items()])
        print(f"Model: {model_name} ({param_str})")
    else:
        print(f"Model: {model_name}")
    
    print("\nDecisions by trial:\n")
    
    # Format the display columns
    display_cols = [
        'Trial', 'Trial Type', 'Total Pot', 
        'Individual Offer', 'Opponent Offer', 'Comparison Offer',
        'Actual D', 'Model D', 'Utility', 'Accept Probability',
        # Add Bayesian debugging metrics
        'Expected Proportion', 'Observed Proportion',
        'Prior mu', 'Delta mu', 'Update Weight',
        'Prior sigma2', 'Prior k', 'Prior v'
    ]
    
    # Only include columns that exist in the results
    display_cols = [col for col in display_cols if col in results.columns]
    
    print(results[display_cols].to_string(index=False))
    
    # ... rest of the display function ...

def plot_utility_curves(total_pot=20, models=None):
    """
    Plot utility curves for different models as the split changes
    
    Parameters:
    total_pot: total amount to be split (default=20)
    models: dictionary of models to plot, if None uses default parameters
    Example models dictionary:
    {
        'Fehr-Schmidt (α=0.9)': {
            'func': fehr_schmidt_utility,
            'params': {'alpha': 0.9, 'beta': 0.25},
            'param_display': 'α=0.9, β=0.25'  # Optional: custom parameter display
        },
        'Custom Model': {
            'func': custom_utility_function,
            'params': {'param1': value1, 'param2': value2},
            'param_display': 'custom display string'  # Optional
        }
    }
    """
    if models is None:
        # Define different values of alpha for Fehr-Schmidt
        alpha_values = [0.1, 0.5, 0.9, 1.5, 2.0]
        models = {
            f'Fehr-Schmidt (α={alpha})': {
                'func': fehr_schmidt_utility,
                'params': {'alpha': alpha, 'beta': 0.25},
                'param_display': f'α={alpha}, β=0.25'
            } for alpha in alpha_values
        }
    
    # Create range of splits to evaluate
    splits = np.linspace(0, total_pot, 100)
    
    # Set up the plot
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Plot each model
    for model_name, model_info in models.items():
        utilities = []
        for own_amount in splits:
            other_amount = total_pot - own_amount
            utility = model_info['func'](
                own_amount=own_amount,
                other_amount=other_amount,
                **model_info['params']
            )
            utilities.append(utility)
        
        plt.plot(splits/total_pot, utilities, label=model_name, linewidth=2)
    
    # Add reference lines
    plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Equal Split')
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Customize plot
    plt.xlabel('Proportion of Pot to Self')
    plt.ylabel('Utility')
    plt.title(f'Utility by Split of {total_pot} Tokens')
    plt.legend()
    
    # Add annotations for parameters
    param_text = []
    for model_name, model_info in models.items():
        if 'param_display' in model_info:
            param_text.append(f"{model_name}: {model_info['param_display']}")
    
    if param_text:
        plt.figtext(0.02, 0.02, '\n'.join(param_text), fontsize=8)
    
    plt.tight_layout()
    plt.show()

# Load data
data = pd.read_csv('cam_made_up_responses.csv')

if __name__ == "__main__":
    # Parameters you can adjust
    participant_id = 1001
    
    # Define all available models and their parameters
    models = {
        'fehr_schmidt': {
            'utility_function': FehrSchmidtModel(0.9, 0.25, 0.001),
            'params': {
                'alpha': 0.9,    # Envy parameter
                'beta': 0.25,    # Guilt parameter
                'temperature': 0.001
            },
            'display_params': ['alpha', 'beta']
        },
        'fehr_schmidt_bayesian': {
            'utility_function': BayesianFehrSchmidtModel(0.9, 0.25, 0.001),
            'params': {
                'alpha': 0.9,           # Envy parameter
                'beta': 0.25,          # Guilt parameter
                'temperature': 0.001,   # Decision temperature
                'initial_k': 1,         # Initial k
                'initial_v': 1,         # Initial v
                'initial_sigma2': 25    # Initial variance
            },
            'display_params': ['alpha', 'beta']
        }
    }
    
    # Run analysis for both models
    for model_name, model_config in models.items():
        utility_model = model_config['utility_function']
        model_params = model_config['params']
        
        # Prepare display parameters
        display_params = {
            param: model_params[param]
            for param in model_config['display_params']
        }
        
        # Analyze decisions
        results = analyze_individual_decisions(
            participant_id, 
            utility_model, 
            data, 
            role='responder'
        )
        
        # Display results
        display_analysis_results(
            results, 
            participant_id, 
            model_name, 
            display_params if display_params else None
        )
        
        # Optional: add a separator between models
        print("\n" + "="*80 + "\n")