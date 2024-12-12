import numpy as np
import pandas as pd
from tqdm import tqdm
import stan
import arviz as az
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BayesianModelFitter:
    def __init__(self, model_name):
        self.model_name = model_name
        self.stan_code = self._get_stan_code()
        
    def _get_stan_code(self):
        """Return Stan code for the specified model"""
        if self.model_name == "FehrSchmidt":
            return """
            data {
                int<lower=0> N;                    // number of trials
                int<lower=0,upper=1> accept[N];    // decisions
                vector[N] offer;                   // offers
                vector[N] total;                   // total amounts
            }
            parameters {
                real<lower=0,upper=10> alpha;      // inequity aversion
                real<lower=0.001,upper=10> temperature;  // softmax temperature
            }
            model {
                // Priors
                alpha ~ normal(2, 2);
                temperature ~ gamma(2, 2);
                
                // Likelihood
                for (i in 1:N) {
                    real utility;
                    real p_accept;
                    
                    utility = offer[i] - alpha * max(total[i] - 2 * offer[i], 0);
                    p_accept = 1 / (1 + exp(-utility / temperature));
                    accept[i] ~ bernoulli(p_accept);
                }
            }
            """
        
        elif self.model_name == "TwoSystem":
            return """
            data {
                int<lower=0> N;
                int<lower=0,upper=1> accept[N];
                vector[N] offer;
                vector[N] total;
            }
            parameters {
                real<lower=0,upper=20> lambda;     // deliberative weight
                real<lower=0,upper=20> tau;        // fairness threshold
                real<lower=0.001,upper=2> temperature;
            }
            model {
                // Priors
                lambda ~ normal(5, 2);
                tau ~ normal(5, 2);
                temperature ~ gamma(2, 2);
                
                // Likelihood
                for (i in 1:N) {
                    real utility;
                    real p_accept;
                    real fairness;
                    real deliberative;
                    
                    fairness = offer[i] / total[i];
                    deliberative = offer[i];
                    utility = lambda * deliberative + (fairness > tau ? 1 : 0);
                    p_accept = 1 / (1 + exp(-utility / temperature));
                    accept[i] ~ bernoulli(p_accept);
                }
            }
            """
            
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

    def prepare_data(self, participant_data):
        """Prepare data for Stan"""
        return {
            'N': len(participant_data),
            'accept': participant_data['accept'].values.astype(int),
            'offer': participant_data['offer'].values,
            'total': participant_data['total'].values
        }

    def fit_participant(self, participant_data, **kwargs):
        """Fit model for a single participant"""
        stan_data = self.prepare_data(participant_data)
        
        # Build and sample
        posterior = stan.build(self.stan_code, data=stan_data)
        fit = posterior.sample(
            num_chains=4,
            num_samples=2000,
            num_warmup=1000,
            **kwargs
        )
        
        # Convert to DataFrame and calculate summary statistics
        df = fit.to_frame()
        
        # Calculate summary statistics based on model
        if self.model_name == "FehrSchmidt":
            summary = self._summarize_fehr_schmidt(df)
        elif self.model_name == "TwoSystem":
            summary = self._summarize_two_system(df)
            
        return summary, df

    def _summarize_fehr_schmidt(self, df):
        """Summarize Fehr-Schmidt model parameters"""
        return {
            'alpha_mean': df['alpha'].mean(),
            'alpha_std': df['alpha'].std(),
            'temperature_mean': df['temperature'].mean(),
            'temperature_std': df['temperature'].std(),
            'alpha_hdi_low': az.hdi(df['alpha'])['lower'],
            'alpha_hdi_high': az.hdi(df['alpha'])['higher'],
            'temperature_hdi_low': az.hdi(df['temperature'])['lower'],
            'temperature_hdi_high': az.hdi(df['temperature'])['higher'],
        }

    def _summarize_two_system(self, df):
        """Summarize Two-System model parameters"""
        return {
            'lambda_mean': df['lambda'].mean(),
            'lambda_std': df['lambda'].std(),
            'tau_mean': df['tau'].mean(),
            'tau_std': df['tau'].std(),
            'temperature_mean': df['temperature'].mean(),
            'temperature_std': df['temperature'].std(),
            'lambda_hdi_low': az.hdi(df['lambda'])['lower'],
            'lambda_hdi_high': az.hdi(df['lambda'])['higher'],
            'tau_hdi_low': az.hdi(df['tau'])['lower'],
            'tau_hdi_high': az.hdi(df['tau'])['higher'],
            'temperature_hdi_low': az.hdi(df['temperature'])['lower'],
            'temperature_hdi_high': az.hdi(df['temperature'])['higher'],
        }

def fit_all_participants(data, model_name, save_individual_traces=False):
    """Fit model for all participants"""
    fitter = BayesianModelFitter(model_name)
    results = []
    
    # Create output directory for traces if needed
    if save_individual_traces:
        trace_dir = Path(f'traces_{model_name.lower()}')
        trace_dir.mkdir(exist_ok=True)
    
    unique_participants = data['ID'].unique()
    
    for participant_id in tqdm(unique_participants, desc="Fitting participants"):
        participant_data = data[data['ID'] == participant_id]
        
        try:
            summary, trace_df = fitter.fit_participant(participant_data)
            
            # Store results
            results.append({
                'participant_id': participant_id,
                **summary
            })
            
            # Save individual traces if requested
            if save_individual_traces:
                trace_df.to_csv(trace_dir / f'participant_{participant_id}_trace.csv')
                
        except Exception as e:
            logging.error(f"Error fitting participant {participant_id}: {str(e)}")
            continue
    
    return pd.DataFrame(results)

def main():
    # Load data
    data_path = Path('../data/ground/UG_all_role1.csv')
    raw_data = pd.read_csv(data_path)
    
    # Models to fit
    models = ["FehrSchmidt", "TwoSystem"]
    
    for model_name in models:
        logging.info(f"Fitting {model_name} model...")
        
        # Fit model
        results_df = fit_all_participants(
            raw_data, 
            model_name,
            save_individual_traces=True
        )
        
        # Save results
        output_path = Path(f'results_{model_name.lower()}_bayesian.csv')
        results_df.to_csv(output_path, index=False)
        
        # Print summary
        logging.info(f"\nFitting Results Summary for {model_name}:")
        logging.info("\n" + str(results_df.describe()))

if __name__ == "__main__":
    main()