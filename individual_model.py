import numpy as np
import pandas as pd

def fehr_schmidt_utility(own_amount, other_amount, alpha, beta):
    """
    Calculate utility based on Fehr-Schmidt inequality aversion model
    
    Parameters:
    own_amount: monetary payoff for self
    other_amount: monetary payoff for other
    alpha: envy parameter (sensitivity to receiving less than other)
    beta: guilt parameter (sensitivity to receiving more than other)
    """
    envy = max(other_amount - own_amount, 0)
    guilt = max(own_amount - other_amount, 0)
    
    utility = own_amount - (alpha * envy) - (beta * guilt)
    return utility

def rabin_utility(own_amount, other_amount, max_possible, min_possible):
    """
    Calculate utility based on Rabin's fairness model
    
    Parameters:
    own_amount: monetary payoff for self (πi)
    other_amount: monetary payoff for other (πj)
    max_possible: maximum possible payoff in the game
    min_possible: minimum possible payoff in the game
    """
    # Calculate fair payoff (average of max and min)
    fair_payoff = (max_possible + min_possible) / 2
    
    # Calculate kindness of player i toward j
    # fi = (πj - πj_fair) / (πj_max - πj_min)
    kindness_to_other = (other_amount - fair_payoff) / (max_possible - min_possible)
    
    # Calculate perceived kindness from j to i
    # f̃j = (πi - πi_fair) / (πi_max - πi_min)
    perceived_kindness = (own_amount - fair_payoff) / (max_possible - min_possible)
    
    # Calculate total utility
    # Ui = πi + f̃j + f̃j * fi
    utility = own_amount + perceived_kindness + (perceived_kindness * kindness_to_other)
    
    return utility

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

def analyze_individual_decisions(participant_id, model_params, data, utility_model, role='responder'):
    """
    Analyze decisions for a specific participant and compare with model predictions
    
    Parameters:
    participant_id: participant ID number
    model_params: dictionary of model parameters
    data: pandas DataFrame containing decision data
    utility_model: function to calculate utility
    role: 'responder' or 'proposer' (default='responder')
    """
    # Filter for participant and role
    individual_data = data[data['ID'] == participant_id].copy()
    if role == 'responder':
        individual_data = individual_data[individual_data['trial_role'] == 1]
    elif role == 'proposer':
        individual_data = individual_data[individual_data['trial_role'] == 2]
    
    temperature = model_params.pop('temperature', 0.001)  # Extract temperature parameter
    
    results = []
    for _, row in individual_data.iterrows():
        # Calculate utility
        if utility_model == rabin_utility:
            utility = utility_model(
                row['split_self'],
                row['split_opp'],
                row['combined_earning'],  # max possible
                0  # min possible
            )
        else:
            utility = utility_model(
                row['split_self'],
                row['split_opp'],
                **model_params
            )
        
        # Simulate decision
        decision, acceptance_prob = simulate_responder_decision(utility, temperature)
        
        results.append({
            'Trial': row['trial_number'],
            'Trial Type': row['trial_type'],
            'Total Pot': row['combined_earning'],
            'Tokens Found (Self)': row['token_self'],
            'Tokens Found (Other)': row['token_opp'],
            'Opponent Offer': row['split_opp'],
            'Individual Offer': row['split_self'],
            'Offer Proportion': row['splitperc_self'] / 100,
            'Actual Decision': 'Accept' if row['accept'] == 1 else 'Reject',
            'Model Decision': 'Accept' if decision else 'Reject',
            'Utility': round(utility, 2),
            'Accept Probability': round(acceptance_prob, 2)
        })
    
    return pd.DataFrame(results)

def display_analysis_results(results, participant_id, alpha, beta):
    """
    Display formatted analysis results for a participant
    
    Parameters:
    results: DataFrame containing analysis results
    participant_id: ID of the participant being analyzed
    alpha: envy parameter used in the analysis
    beta: guilt parameter used in the analysis
    """
    print(f"\nAnalysis for Participant {participant_id} (α={alpha}, β={beta}):\n")
    print("Decisions by trial:\n")
    
    # Set display options for cleaner output
    pd.set_option('display.max_rows', None)
    pd.set_option('display.float_format', lambda x: '{:.2f}'.format(x))
    
    # Create a cleaner display table
    display_results = results.sort_values('Trial')[
        ['Trial Type', 'Tokens Found (Self)', 'Tokens Found (Other)',
         'Individual Offer', 'Opponent Offer', 
         'Actual Decision', 'Model Decision', 
         'Utility', 'Accept Probability']
    ].rename(columns={
        'Accept Probability': 'P(Accept)'
    })
    
    # Center-align the decision columns
    for col in ['Actual Decision', 'Model Decision']:
        display_results[col] = display_results[col].str.center(10)
    
    # Print the results with additional spacing
    print(display_results.to_string(index=False, col_space=12, justify='right'))
    print("\n" + "-" * 60 + "\n")
    
    # Calculate and display statistics
    print("Summary Statistics:\n")
    agreement = (results['Actual Decision'] == results['Model Decision']).mean()
    actual_acceptance_rate = (results['Actual Decision'] == 'Accept').mean()
    model_acceptance_rate = (results['Model Decision'] == 'Accept').mean()
    
    print(f"Model agreement with actual decisions: {agreement:.1%}\n")
    print(f"Actual acceptance rate: {actual_acceptance_rate:.1%}\n")
    print(f"Model acceptance rate: {model_acceptance_rate:.1%}\n")

# Load data
data = pd.read_csv('cam_made_up_responses.csv')

if __name__ == "__main__":
    # Parameters you can adjust
    participant_id = 1001  # Updated to match new dataset
    
    # Choose your model and its parameters
    model_params = {
        'alpha': 0.9,  # Envy parameter
        'beta': 0.25   # Guilt parameter
    }
    
    # Select utility model (currently only Fehr-Schmidt implemented)
    utility_model = rabin_utility
    
    # Analyze decisions
    results = analyze_individual_decisions(
        participant_id, 
        model_params, 
        data, 
        utility_model,
        role='responder'  # Specify role
    )
    
    # Display results
    display_analysis_results(results, participant_id, model_params['alpha'], model_params['beta'])