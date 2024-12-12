import numpy as np

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
        # guilt = max(own_amount - other_amount, 0) ignored because not represented by data
        return own_amount - (self.alpha * envy) #  - (self.beta * guilt)
    
    def calculate_trial_utility(self, row):
        utility = self.calculate_utility(row['split_self'], row['split_opp'])
        return {
            'utility': utility,
            'comparison_offer': row['split_self'],
            'additional_metrics': {}
        }

class BayesianFehrSchmidtModel(UtilityModel):
    def __init__(self, alpha, beta, temperature, initial_k=1):
        super().__init__(alpha, beta, temperature)
        self.mu_hat = 0.5  # Initial expected proportion
        self.k = initial_k

    def calculate_utility(self, own_amount, expected_offer):
        envy = max(expected_offer - own_amount, 0)
        # guilt = max(own_amount - expected_offer, 0)
        return own_amount - (self.alpha * envy) # - (self.beta * guilt)
    
    def calculate_trial_utility(self, row):
        total_pot = row['combined_earning']
        current_mu = self.mu_hat  # Store current mu_hat before update
        expected_offer = current_mu * total_pot
        
        utility = self.calculate_utility(row['split_self'], expected_offer)
        
        # Calculate observed proportion
        observed_proportion = row['split_self'] / total_pot
        
        # Store previous values for debugging
        prev_k = self.k
        
        # Perform Bayesian update
        self.mu_hat, self.k = bayesian_update(
            observed_proportion,
            current_mu,
            prev_k
        )
        
        return {
            'utility': utility,
            'comparison_offer': expected_offer,
            'additional_metrics': {
                'Expected Proportion': round(current_mu, 2),
                'Observed Proportion': round(observed_proportion, 2),
                'Prior mu': round(current_mu, 2),
                'Prior k': round(prev_k, 2),
                'Delta mu': round(self.mu_hat - current_mu, 4),
                'Update Weight': round(1/self.k, 4),
                'Expected Offer': round(expected_offer, 2),
                'Total Pot': total_pot
            }
        }

class TwoSystemModel(UtilityModel):
    """
    Implementation of the emotional utility model for the Ultimatum Game
    as described in the paper. The utility is calculated as:
    u_i(x) = x_i + ε(x_i, λ, τ)
    where ε(x_i, λ, τ) = v(x) * a(x_i, λ, τ)
    """
    def __init__(self, lambda_param, tau, temperature):
        """
        Initialize the emotional utility model.
        
        Parameters:
        lambda_param: Intensity of emotional influence (0 ≤ λ ≤ 1)
        tau: Threshold for emotional activation (0 ≤ τ ≤ 1)
        temperature: Softmax temperature parameter for decision making
        """
        super().__init__(lambda_param, tau, temperature)
        self.lambda_param = lambda_param
        self.tau = tau
        
    def calculate_valence(self, proportion):
        """Calculate the valence v(x) based on deviation from equal split"""
        if proportion < 0.5:
            return -1
        elif proportion > 0.5:
            return 1
        return 0
        
    def calculate_arousal(self, proportion):
        """
        Calculate the arousal a(x_i, λ, τ) using Heaviside function
        Returns λ if deviation from equal split exceeds τ, 0 otherwise
        """
        deviation = abs(2 * proportion - 1)
        if deviation > self.tau:
            return self.lambda_param
        return 0
        
    def calculate_emotional_component(self, proportion):
        """Calculate the emotional component ε(x_i, λ, τ)"""
        valence = self.calculate_valence(proportion)
        arousal = self.calculate_arousal(proportion)
        return valence * arousal
        
    def calculate_utility(self, own_amount, other_amount):
        """
        Calculate total utility including both monetary and emotional components
        
        Parameters:
        own_amount: Amount offered to self
        other_amount: Amount offered to other player
        """
        total_amount = own_amount + other_amount
        proportion = own_amount / total_amount

        # Add emotional component
        emotional_utility = self.calculate_emotional_component(proportion)
        
        #return own_amount + emotional_utility
        return 1 + emotional_utility
    
    def calculate_trial_utility(self, row):
        """
        Calculate utility for a specific trial and return relevant metrics
        
        Parameters:
        row: DataFrame row containing trial data
        """
        total_pot = row['combined_earning']
        own_amount = row['split_self']
        
        proportion = own_amount / total_pot 
        
        # Calculate component
        emotional_component = self.calculate_emotional_component(proportion)
        total_utility = own_amount + emotional_component
        
        return {
            'utility': total_utility,
            'comparison_offer': own_amount,
            'additional_metrics': {
                'Proportion': round(proportion, 2),
                'Valence': self.calculate_valence(proportion),
                'Arousal': round(self.calculate_arousal(proportion), 2),
                'Emotional Component': round(emotional_component, 2),
                'Monetary Utility': round(own_amount, 2)
            }
        }
    
class TwoSystemBayesianModel(UtilityModel):
    def __init__(self, lambda_param, tau, temperature, initial_k=1):
        super().__init__(lambda_param, tau, temperature)
        self.lambda_param = lambda_param
        self.tau = tau
        self.mu_hat = 0.5  # Initial expected proportion
        self.k = initial_k

    def calculate_valence(self, proportion, expected_proportion):
        """
        Calculate the valence v(x) based on deviation from expected proportion
        Returns:
        -1 if received less than expected
        1 if received more than expected
        0 if received exactly what was expected
        """
        if proportion < expected_proportion:
            return -1
        elif proportion > expected_proportion:
            return 1
        return 0
        
    def calculate_arousal(self, proportion, expected_proportion):
        """
        Returns λ if deviation from expected proportion exceeds τ, 0 otherwise
        """
        deviation = abs(proportion - expected_proportion)
        if deviation > self.tau:
            return self.lambda_param
        return 0
        
    def calculate_emotional_component(self, proportion, expected_proportion):
        """Calculate the emotional component ε(x_i, λ, τ)"""
        valence = self.calculate_valence(proportion, expected_proportion)
        arousal = self.calculate_arousal(proportion, expected_proportion)
        return valence * arousal
        
    def calculate_utility(self, own_amount, total_amount):
        """Calculate total utility including both monetary and emotional components"""
        proportion = own_amount / total_amount
        expected_proportion = self.mu_hat  # Use Bayesian expected proportion
        
        # Add emotional component
        emotional_utility = self.calculate_emotional_component(proportion, expected_proportion)
        
        return own_amount + emotional_utility
    
    def calculate_trial_utility(self, row):
        """Calculate utility for a specific trial and return relevant metrics"""
        total_pot = row['combined_earning']
        own_amount = row['split_self']
        current_mu = self.mu_hat  # Store current mu_hat before update
        
        proportion = own_amount / total_pot
        
        # Calculate components using current expectations
        emotional_component = self.calculate_emotional_component(proportion, current_mu)
        total_utility = own_amount + emotional_component
        
        # Store previous values for debugging
        prev_k = self.k
        
        # Perform Bayesian update
        self.mu_hat, self.k = bayesian_update(
            proportion,  # observed proportion
            current_mu,
            prev_k
        )
        
        return {
            'utility': total_utility,
            'comparison_offer': own_amount,
            'additional_metrics': {
                'Proportion': round(proportion, 2),
                'Expected Proportion': round(current_mu, 2),
                'Valence': self.calculate_valence(proportion, current_mu),
                'Arousal': round(self.calculate_arousal(proportion, current_mu), 2),
                'Emotional Component': round(emotional_component, 2),
                'Monetary Utility': round(own_amount, 2),
                'Prior mu': round(current_mu, 2),
                'Prior k': round(prev_k, 2),
                'Delta mu': round(self.mu_hat - current_mu, 4),
                'Update Weight': round(1/self.k, 4)
            }
        }

class RWFehrSchmidtModel(UtilityModel):
    def __init__(self, alpha, beta, temperature, learning_rate=0.3):
        super().__init__(alpha, beta, temperature)
        self.learning_rate = learning_rate
        self.expected_proportion = 0.5  # Initial expectation
        
    def calculate_utility(self, own_amount, expected_offer):
        envy = max(expected_offer - own_amount, 0)
        # guilt = max(own_amount - expected_offer, 0)
        return own_amount - (self.alpha * envy) #  - (self.beta * guilt)
    
    def calculate_trial_utility(self, row):
        total_pot = row['combined_earning']
        current_expectation = self.expected_proportion  # Store current expectation
        expected_offer = current_expectation * total_pot
        
        utility = self.calculate_utility(row['split_self'], expected_offer)
        
        # Calculate observed proportion
        observed_proportion = row['split_self'] / total_pot
        
        # Perform RW update
        self.expected_proportion = rescorla_wagner_update(
            current_expectation,
            observed_proportion,
            self.learning_rate
        )
        
        return {
            'utility': utility,
            'comparison_offer': expected_offer,
            'additional_metrics': {
                'Expected Proportion': round(current_expectation, 2),
                'Observed Proportion': round(observed_proportion, 2),
                'Prior Expectation': round(current_expectation, 2),
                'Delta Expectation': round(self.expected_proportion - current_expectation, 4),
                'Learning Rate': self.learning_rate,
                'Expected Offer': round(expected_offer, 2),
                'Total Pot': total_pot
            }
        }

def bayesian_update(x_t_proportion, mu_hat_prev, k_prev):
    """
    Perform Bayesian update for the mean of the offer proportions.
    
    Parameters:
    x_t_proportion: observed offer proportion
    mu_hat_prev: previous estimated mean proportion μ̂_{t-1}
    k_prev: previous k value

    Returns:
    mu_hat_t: updated estimated mean proportion μ̂_t
    k_t: updated k_t
    """
    k_t = k_prev + 1
    
    # Update mean estimate (weighted average of previous estimate and new observation)
    mu_hat_t = (k_prev / k_t) * mu_hat_prev + (1 / k_t) * x_t_proportion
    
    return mu_hat_t, k_t

def rescorla_wagner_update(prediction, outcome, learning_rate):
    """
    Perform Rescorla-Wagner update for expected value.
    
    Parameters:
    prediction: current prediction/expectation
    outcome: observed outcome
    learning_rate: learning rate parameter (α) controlling update speed
    
    Returns:
    updated prediction
    """
    prediction_error = outcome - prediction
    new_prediction = prediction + learning_rate * prediction_error
    return new_prediction

class PropertyFehrSchmidtModel(UtilityModel):
    def calculate_utility(self, own_amount, other_amount, token_self, token_opp):
        envy = max(other_amount - own_amount, 0)
        rho = token_self / token_opp
        return own_amount - (self.alpha * envy * rho)
    
    def calculate_trial_utility(self, row):
        utility = self.calculate_utility(
            row['split_self'], 
            row['split_opp'],
            row['token_self'],
            row['token_opp']
        )
        return {
            'utility': utility,
            'comparison_offer': row['split_self'],
            'additional_metrics': {
                'rho': round(row['token_self'] / row['token_opp'], 4)
            }
        }

class PropertyBayesianFehrSchmidtModel(UtilityModel):
    def __init__(self, alpha, beta, temperature, initial_k=1):
        super().__init__(alpha, beta, temperature)
        self.mu_hat = 0.5
        self.k = initial_k

    def calculate_utility(self, own_amount, expected_offer, token_self, token_opp):
        envy = max(expected_offer - own_amount, 0)
        rho = token_self / token_opp
        return own_amount - (self.alpha * envy * rho)
    
    def calculate_trial_utility(self, row):
        total_pot = row['combined_earning']
        current_mu = self.mu_hat
        expected_offer = current_mu * total_pot
        
        utility = self.calculate_utility(
            row['split_self'], 
            expected_offer,
            row['token_self'],
            row['token_opp']
        )
        
        observed_proportion = row['split_self'] / total_pot
        prev_k = self.k
        
        self.mu_hat, self.k = bayesian_update(
            observed_proportion,
            current_mu,
            prev_k
        )
        
        return {
            'utility': utility,
            'comparison_offer': expected_offer,
            'additional_metrics': {
                'rho': round(row['token_self'] / row['token_opp'], 4),
                'Expected Proportion': round(current_mu, 2),
                'Observed Proportion': round(observed_proportion, 2),
                'Prior mu': round(current_mu, 2),
                'Prior k': round(prev_k, 2),
                'Delta mu': round(self.mu_hat - current_mu, 4),
                'Update Weight': round(1/self.k, 4),
                'Expected Offer': round(expected_offer, 2),
                'Total Pot': total_pot
            }
        }

class PropertyRWFehrSchmidtModel(UtilityModel):
    def __init__(self, alpha, beta, temperature, learning_rate=0.3):
        super().__init__(alpha, beta, temperature)
        self.learning_rate = learning_rate
        self.expected_proportion = 0.5
        
    def calculate_utility(self, own_amount, expected_offer, token_self, token_opp):
        envy = max(expected_offer - own_amount, 0)
        rho = token_self / token_opp
        return own_amount - (self.alpha * envy * rho)
    
    def calculate_trial_utility(self, row):
        total_pot = row['combined_earning']
        current_expectation = self.expected_proportion
        expected_offer = current_expectation * total_pot
        
        utility = self.calculate_utility(
            row['split_self'], 
            expected_offer,
            row['token_self'],
            row['token_opp']
        )
        
        observed_proportion = row['split_self'] / total_pot
        
        self.expected_proportion = rescorla_wagner_update(
            current_expectation,
            observed_proportion,
            self.learning_rate
        )
        
        return {
            'utility': utility,
            'comparison_offer': expected_offer,
            'additional_metrics': {
                'rho': round(row['token_self'] / row['token_opp'], 4),
                'Expected Proportion': round(current_expectation, 2),
                'Observed Proportion': round(observed_proportion, 2),
                'Prior Expectation': round(current_expectation, 2),
                'Delta Expectation': round(self.expected_proportion - current_expectation, 4),
                'Learning Rate': self.learning_rate,
                'Expected Offer': round(expected_offer, 2),
                'Total Pot': total_pot
            }
        }
