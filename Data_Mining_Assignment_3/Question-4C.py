from scipy.special import comb

def majority_vote_accuracy(n_models, model_accuracy):
    
    # Number of models required for a majority
    majority = n_models // 2 + 1
    
    # Total probability for a majority vote
    total_probability = sum(comb(n_models, k) * (model_accuracy**k) * ((1-model_accuracy)**(n_models-k)) for k in range(majority, n_models+1))
    return total_probability

# Parameters
n_models = 25
model_accuracy = 0.6 # 60% accuracy

# Calculate the accuracy of the majority vote classifier
majority_vote_acc = majority_vote_accuracy(n_models, model_accuracy)
print(f"The accuracy of the majority vote classifier C25 is approximately {majority_vote_acc:.4f}")
