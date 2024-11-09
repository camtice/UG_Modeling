import numpy as np
import pandas as pd

def calculate_utility(own_amount, other_amount, alpha, beta):
    """
    Fehr-Schmidt inequality aversion model
    """
    envy = max(other_amount - own_amount, 0)
    guilt = max(own_amount - other_amount, 0)
    
    utility = own_amount - (alpha * envy) - (beta * guilt)
    return utility

def simulate_responder_decision(own_amount, other_amount, alpha=0.5, beta=0.25):
    """
    Simulate responder's decision to accept or reject an offer
    """
    utility_accept = calculate_utility(own_amount, other_amount, alpha, beta)
    utility_reject = calculate_utility(0, 0, alpha, beta)
    return utility_accept > utility_reject

# Load empirical data
empirical_data = pd.read_csv('/Users/camerontice/Desktop/MPhil/UG_Modeling/ultimatum_game_responses_v1.csv')

def generate_simulated_data(empirical_data, n_participants=10):
    """
    Generate simulated data matching the structure of the empirical dataset
    """
    simulated_data = []
    
    # Generate random parameters for different participants
    alphas = np.random.uniform(0.3, 0.7, n_participants)
    betas = np.random.uniform(0.1, 0.4, n_participants)
    
    for participant in range(1, n_participants + 1):
        alpha = alphas[participant - 1]
        beta = betas[participant - 1]
        
        # Get all unique game configurations from empirical data
        game_configs = empirical_data[['Total Pot', 'Individual Offer', 'Opponent Offer']].drop_duplicates().values
        
        for offer_num, (total_pot, own_amount, other_amount) in enumerate(game_configs, 1):
            decision = simulate_responder_decision(own_amount, other_amount, alpha, beta)
            
            simulated_data.append({
                'Participant ID': participant,
                'Offer Number': offer_num,
                'Total Pot': total_pot,
                'Individual Offer': own_amount,
                'Opponent Offer': other_amount,
                'Decision': 1 if decision else 0,
                'Alpha': alpha,
                'Beta': beta
            })
    
    return pd.DataFrame(simulated_data)

# Generate simulated data
simulated_data = generate_simulated_data(empirical_data)

# Save simulated data
simulated_data.to_csv('simulated_ultimatum_game.csv', index=False)

# Print summary statistics
print("\nSimulation Summary:")
print(f"Average acceptance rate: {simulated_data['Decision'].mean():.2%}")
print(f"Empirical acceptance rate: {empirical_data['Decision'].mean():.2%}")

# Compare acceptance rates by offer proportion
simulated_data['Offer Proportion'] = simulated_data['Individual Offer'] / simulated_data['Total Pot']
empirical_data['Offer Proportion'] = empirical_data['Individual Offer'] / empirical_data['Total Pot']

print("\nAcceptance rates by offer proportion quartiles:")
print("\nSimulated:")
simulated_data['Offer Quartile'] = pd.qcut(simulated_data['Offer Proportion'], 4)
print(simulated_data.groupby('Offer Quartile')['Decision'].mean())

print("\nEmpirical:")
empirical_data['Offer Quartile'] = pd.qcut(empirical_data['Offer Proportion'], 4)
print(empirical_data.groupby('Offer Quartile')['Decision'].mean())
