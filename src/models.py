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
        
        return own_amount + emotional_utility
    
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
    def __init__(self, lambda_param, tau, temperature, initial_k=1, initial_v=1, initial_sigma2=0.04):
        super().__init__(lambda_param, tau, temperature)
        self.lambda_param = lambda_param
        self.tau = tau
        self.mu_hat = 0.5  # Initial expected proportion
        self.k = initial_k
        self.v = initial_v
        self.sigma2_hat = initial_sigma2
        
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
        Calculate the arousal using Heaviside function and expected proportion
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
        prev_v = self.v
        prev_sigma2 = self.sigma2_hat
        
        # Perform Bayesian update
        self.mu_hat, self.k, self.v, self.sigma2_hat = bayesian_update(
            proportion,  # observed proportion
            current_mu,
            prev_k,
            prev_v,
            prev_sigma2
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
                'Variance': round(self.sigma2_hat, 4),
                'Prior mu': round(current_mu, 2),
                'Prior k': round(prev_k, 2),
                'Prior v': round(prev_v, 2),
                'Prior sigma2': round(prev_sigma2, 4),
                'Delta mu': round(self.mu_hat - current_mu, 4),
                'Update Weight': round(1/self.k, 4)
            }
        }


def bayesian_update(x_t_proportion, mu_hat_prev, k_prev, v_prev, sigma2_hat_prev):
    """
    Perform Bayesian update for the mean and variance of the offer proportions.

    Parameters:
    x_t_proportion: observed offer proportion
    mu_hat_prev: previous estimated mean proportion μ̂_{t-1}

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