import pandas as pd

# Create systematic test data
data = []

# Offer percentages from 60% down to 10% in steps of 10
offer_percentages = [0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
pot_sizes = [9, 12, 15]

# Generate 3 variations for each combination
for pot in pot_sizes:
    for offer_pct in offer_percentages:
        offer_amount = round(pot * offer_pct, 2)
        # Create 3 variations with slightly different combined earnings
        for variation in range(3):
            base_earnings = pot - offer_amount + offer_amount  # Combined earnings if accepted
            # Vary the combined earnings slightly for each variation
            earnings_variation = base_earnings * (1 + (variation - 1) * 0.05)
            
            data.append({
                'trial_role': 1,  # Responder
                'pot_size': pot,
                'offer_amount': offer_amount,
                'offer_percentage': offer_pct * 100,
                'combined_earnings': round(earnings_variation, 2)
            })

# Create DataFrame
df = pd.DataFrame(data)

# Sort by pot size and offer percentage for clarity
df = df.sort_values(['pot_size', 'offer_percentage'], ascending=[True, False])

# Display the data
print(df.to_string(index=False))
