import numpy as np
import pandas as pd
from datetime import datetime  # Import datetime for timestamp
import os  # Import os for directory handling
from models import FehrSchmidtModel, BayesianFehrSchmidtModel


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
    """Analyze decisions for a specific participant using model predictions"""
    # Filter for participant and role
    individual_data = data[data['ID'] == participant_id].copy()
    
    # Remove the role filtering if you want to analyze all trials
    individual_data = individual_data[individual_data['trial_role'] == (1 if role == 'responder' else 2)]
    
    results = []
    for _, row in individual_data.iterrows():
        # Calculate utility using model-specific logic
        trial_result = utility_model.calculate_trial_utility(row)
        utility = trial_result['utility']
        comparison_offer = trial_result['comparison_offer']
        
        # Simulate new decision (independent of original data)
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
            'Model Decision': 'Accept' if decision else 'Reject',
            'Utility': round(utility, 2),
            'Accept Probability': round(acceptance_prob, 2)
        }
        
        # Add any additional model-specific metrics
        result_dict.update(trial_result['additional_metrics'])
        results.append(result_dict)
    
    return pd.DataFrame(results)


def display_analysis_results(results_dict, model_name, model_params=None):
    """Display and save results for a specific model across all participants"""
    print(f"\nAnalysis for {model_name}")
    if model_params:
        param_str = ", ".join([f"{k}={v}" for k, v in model_params.items()])
        print(f"Model Parameters: {param_str}")
    
    # Combine all participants' results into one DataFrame
    all_results = []
    for participant_id, results in results_dict.items():
        results_df = results.copy()
        results_df['Participant ID'] = participant_id
        all_results.append(results_df)
    
    combined_results = pd.concat(all_results, ignore_index=True)
    
    # Base columns with Participant ID first
    base_cols = [
        'Participant ID', 'Trial', 'Total Pot', 
        'Individual Offer', 'Opponent Offer',
        'Model Decision', 'Utility', 'Accept Probability'
    ]
    
    # Add Bayesian-specific columns if they exist
    bayesian_cols = [
        'Expected Proportion', 'Variance'
    ]
    
    # Add emotional model columns if they exist
    emotional_cols = [
        'Valence', 'Arousal'
    ]
    
    # Combine all possible columns
    all_display_cols = base_cols + bayesian_cols + emotional_cols
    
    # Only include columns that exist in the results
    display_cols = [col for col in all_display_cols if col in combined_results.columns]
    
    # Print the results
    print(combined_results[display_cols].to_string(index=False))
    
    # Create the directory if it doesn't exist
    output_dir = 'data/simulated'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results to CSV with timestamp
    timestamp = datetime.now().strftime("%m%d_%H%M")
    filename = f"{output_dir}/{model_name}_results_{timestamp}.csv"
    combined_results[display_cols].to_csv(filename, index=False)
    print(f"\nResults saved to {filename}")


def run_analysis(data, participant_ids=None, models_config=None):
    """Main function to run analysis and return results
    
    Parameters:
    data: DataFrame containing all participant data
    participant_ids: List of participant IDs or None (for all participants)
    models_config: Dictionary of model configurations
    """
    if participant_ids is None:
        # Get all unique participant IDs from the data
        participant_ids = data['ID'].unique()
    elif isinstance(participant_ids, (int, str)):
        # Convert single participant ID to list
        participant_ids = [participant_ids]
        
    all_results = {}
    for participant_id in participant_ids:
        participant_results = {}
        for model_name, model_config in models_config.items():
            # Create a fresh instance of the model for each participant
            model_class = model_config['model_class']
            model_params = model_config['params']
            utility_model = model_class(**model_params)
            
            model_results = analyze_individual_decisions(
                participant_id, 
                utility_model, 
                data, 
                role='responder'
            )
            participant_results[model_name] = model_results
        all_results[participant_id] = participant_results
    
    return all_results