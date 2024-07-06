import streamlit as st
import pickle

# Load your decision tree model
with open('decision_tree_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Define the branch preference order for adjustment
branch_preference_order = ['CSE', 'IT', 'CSE(AIML)', 'CSE(DS)']

# Function to adjust probabilities based on preference order
def adjust_probabilities(probabilities):
    max_prob_index = None
    for i, branch in enumerate(branch_preference_order):
        if probabilities[branch] == 1.0:
            max_prob_index = i
            break
    if max_prob_index is not None:
        for j in range(max_prob_index + 1, len(branch_preference_order)):
            probabilities[branch_preference_order[j]] = 1.0
    return probabilities

# Define the Streamlit app
def main():
    st.title('Branch Upgrade Prediction')

    # Input YGPA from user
    ygpa = st.number_input('Enter your YGPA:', min_value=0.0, max_value=10.0, step=0.01, format="%.2f")

    if st.button('Predict'):
        # Validate input
        if ygpa < 8.5:
            st.error('Minimum YGPA requirement not met')
        else:
            # Predict the best branch and its probability
            predicted_branch = model.predict([[ygpa]])[0]
            probabilities = model.predict_proba([[ygpa]])[0]
            branch_probabilities = {branch: prob for branch, prob in zip(model.classes_, probabilities)}

            # Adjust probabilities based on preference order
            branch_probabilities = adjust_probabilities(branch_probabilities)

            # Display results for the selected branches
            st.write(f'Predicted Branch: {predicted_branch}')
            st.write('Probabilities:')
            for branch in branch_preference_order:
                st.write(f'{branch}: {branch_probabilities.get(branch, 0.0)}')

if __name__ == '__main__':
    main()
