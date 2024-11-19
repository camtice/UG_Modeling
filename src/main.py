import pandas as pd
from models import FehrSchmidtModel, BayesianFehrSchmidtModel, TwoSystemModel, TwoSystemBayesianModel
from processing import run_analysis, display_analysis_results
from visualize import plot_utility_curves, create_utility_animation

def main():
    # Update path to be relative to project root
    data = pd.read_csv('./data/ground/UG_raw_data_with_trials.csv')
    
    # Define models configuration in one place
    model_params = {
        'fehr_schmidt': {
            'alpha': 0.9,
            'beta': 0.25,
            'temperature': 1
        },
        'fehr_schmidt_bayesian': {
            'alpha': 5,
            'beta': 0.25,
            'temperature': 1,
            'initial_k': 8,
            'initial_v': 8,
            'initial_sigma2': 0.04
        },
        'two_system': {
            'lambda_param': 2,
            'tau': 0.0,
            'temperature': 1
        },
        'two_system_bayesian': {
            'lambda_param': 4,
            'tau': 0.0,
            'temperature': 1,
            'initial_k': 8,
            'initial_v': 8,
            'initial_sigma2': 0.04
        }
    }
    
    # Create models configuration
    models_config = {
        'fehr_schmidt': {
            'model_class': FehrSchmidtModel,
            'params': model_params['fehr_schmidt'],
            'display_params': ['alpha', 'beta']
        },
        'fehr_schmidt_bayesian': {
            'model_class': BayesianFehrSchmidtModel,
            'params': model_params['fehr_schmidt_bayesian'],
            'display_params': ['alpha', 'beta']
        },
        'two_system': {
            'model_class': TwoSystemModel,
            'params': model_params['two_system'],
            'display_params': ['lambda_param', 'tau']
        },
        'two_system_bayesian': {
            'model_class': TwoSystemBayesianModel,
            'params': model_params['two_system_bayesian'],
            'display_params': ['lambda_param', 'tau']
        }
    }
    
    # Run analysis for all participants (or specify a list)
    participant_ids = [1001]  # Specific participants
    # participant_ids = None  # All participants
    results = run_analysis(data, participant_ids, models_config)
    
    # Reorganize results by model instead of by participant
    model_results = {}
    for participant_id, participant_results in results.items():
        for model_name, model_results_df in participant_results.items():
            if model_name not in model_results:
                model_results[model_name] = {}
            model_results[model_name][participant_id] = model_results_df

    # Display results grouped by model
    for model_name, participant_results in model_results.items():
        print(f"\n{'='*50}")
        print(f"Results for Model: {model_name}")
        print(f"{'='*50}")
        display_analysis_results(
            participant_results,
            model_name,
            models_config[model_name].get('params')
        )
    
    # Create visualizations
    # plot_utility_curves(total_pot=20)
    # create_utility_animation(total_pot=20)

if __name__ == "__main__":
    main()