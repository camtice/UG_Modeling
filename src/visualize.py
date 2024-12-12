import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.animation as animation
from models import FehrSchmidtModel, BayesianFehrSchmidtModel

def plot_utility_curves(total_pot=20, models=None):
    """Plot utility curves comparing Fehr-Schmidt and Bayesian Fehr-Schmidt models"""
    if models is None:
        # Create both regular and Bayesian models
        alpha = 2
        beta = 0.25
        models = {
            'Fehr-Schmidt': {
                'model': FehrSchmidtModel(alpha=alpha, beta=beta, temperature=0.001),
                'param_display': f'α={alpha}, β={beta}'
            },
            'Bayesian Fehr-Schmidt': {
                'model': BayesianFehrSchmidtModel(alpha=alpha, beta=beta, temperature=0.001),
                'param_display': f'α={alpha}, β={beta}, μ₀=0.5'
            }
        }
    
    # Create range of splits to evaluate
    splits = np.linspace(0, total_pot, 100)
    
    # Set up the plot
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    
    # Plot regular Fehr-Schmidt model
    utilities = []
    for own_amount in splits:
        other_amount = total_pot - own_amount
        utility = models['Fehr-Schmidt']['model'].calculate_utility(own_amount, other_amount)
        utilities.append(utility)
    plt.plot(splits/total_pot, utilities, label='Fehr-Schmidt', linewidth=2, color='black')
    
    # Plot Bayesian model with different expected proportions
    expected_proportions = [0.5, 0.4, 0.3, 0.2]
    colors = ['red', 'orange', 'green', 'blue']
    
    for exp_prop, color in zip(expected_proportions, colors):
        utilities = []
        models['Bayesian Fehr-Schmidt']['model'].mu_hat = exp_prop
        for own_amount in splits:
            utility = models['Bayesian Fehr-Schmidt']['model'].calculate_utility(
                own_amount, exp_prop * total_pot)
            utilities.append(utility)
        plt.plot(splits/total_pot, utilities, 
                label=f'Bayesian FS (exp={exp_prop})', 
                linewidth=2, 
                color=color)
        
        # Add vertical line for each expected proportion
        plt.axvline(x=exp_prop, color=color, linestyle=':', alpha=0.5)
    
    # Add reference lines
    plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Equal Split')
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Customize plot
    plt.xlabel('Proportion of Pot to Self')
    plt.ylabel('Utility')
    plt.title('Utility Curves: Fehr-Schmidt vs Bayesian Fehr-Schmidt\nDotted lines show expected proportions')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add annotations for parameters
    param_text = []
    for model_name, model_info in models.items():
        if 'param_display' in model_info:
            param_text.append(f"{model_name}: {model_info['param_display']}")
    
    if param_text:
        plt.figtext(0.02, 0.02, '\n'.join(param_text), fontsize=8)
    
    plt.tight_layout()
    plt.show()


def create_utility_animation(total_pot=20, frames=50):
    """Create an animated plot showing utility curves changing with expectations"""
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Create range of splits to evaluate
    splits = np.linspace(0, total_pot, 100)
    
    # Define different alpha (envy) values
    alphas = [0.5, 1.0, 2.0]
    colors = ['blue', 'green', 'red']
    
    def animate(frame):
        ax.clear()
        sns.set_style("whitegrid")
        
        # Calculate expected proportion (oscillating between 0.2 and 0.5)
        t = frame / frames
        expected_prop = 0.35 + 0.15 * np.sin(2 * np.pi * t)
        
        # Plot for each alpha value
        for alpha, color in zip(alphas, colors):
            # Create models
            bayes_model = BayesianFehrSchmidtModel(
                alpha=alpha, 
                beta=0.25, 
                temperature=0.001
            )
            bayes_model.mu_hat = expected_prop
            
            # Calculate utilities
            utilities = []
            for own_amount in splits:
                utility = bayes_model.calculate_utility(
                    own_amount, 
                    expected_prop * total_pot
                )
                utilities.append(utility)
            
            ax.plot(splits/total_pot, utilities, 
                   label=f'α={alpha}', 
                   linewidth=2, 
                   color=color)
        
        # Add reference lines
        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Equal Split')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=expected_prop, color='black', linestyle=':', alpha=0.5,
                  label=f'Expected Prop: {expected_prop:.2f}')
        
        # Customize plot
        ax.set_xlabel('Proportion of Pot to Self')
        ax.set_ylabel('Utility')
        ax.set_title(f'Bayesian Fehr-Schmidt Utility Curves\nExpected Proportion = {expected_prop:.2f}')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_ylim(-15, 20)  # Adjust these values based on your utility ranges
        
    # Create animation
    anim = animation.FuncAnimation(
        fig, 
        animate, 
        frames=frames,
        interval=100,  # 100ms between frames
        repeat=True
    )
    
    # Save animation
    anim.save('utility_animation.gif', writer='pillow')
    plt.close()
