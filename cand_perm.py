from itertools import product

# Only top from categories
categories = {
    "Intake": {"Funnel": 0.275, "Chute": 0.232, "Auger": 0.217, "Vibratory Bowl": 0.185, "Front Loader": 0.0916},
    "Deploy": {"R-R": 0.262, "SR-CL": 0.167, "SR-WP": 0.150, "SR-CB": 0.136, "PR-CL": 0.114, "PR-WP": 0.0926, "PR-CB": 0.0790},
    "Interface": {"Magnetic": 0.226, "Flatbed": 0.215, "Coupling": 0.150, "Gecko": 0.107, "Compliant Gripper": 0.105, "Rigid Gripper": 0.104, "Hook": 0.0945},
    "Transport": {"Forklift": 0.213, "SCARA": 0.218, "4-DoF": 0.236, "Crane": 0.171, "5-DoF": 0.161}
    # "Transport": {"Forklift": 0.204, "SCARA": 0.235, "4-DoF": 0.233, "Crane": 0.184, "5-DoF": 0.144},
}

n_total = 24

n_intake = 5
n_options = {
    "Intake": 5,
    "Deploy": 7,
    "Interface": 7,
    "Transport": 5
}
# Adjusted preferences (scaled by n_category / n_total)
adjusted_categories = {}
global_sum = 0

for category, prefs in categories.items():
    scale = n_options[category] / n_total
    adjusted_categories[category] = {
        name: value * scale for name, value in prefs.items()
    }
    global_sum += sum(adjusted_categories[category].values())

# Output adjusted preferences
# for category, prefs in adjusted_categories.items():
#     print(f"{category}:")
#     for name, value in prefs.items():
#         print(f"  {name}: {value:.4f}")

# print(f"\nTotal sum across all categories: {global_sum:.6f}")

categories = adjusted_categories

# All data

def calculate_combined_scores(categories):
    # Extract category names and candidate sets
    category_names = list(categories.keys())
    candidate_options = [list(categories[category].items()) for category in category_names]

    # Generate all combinations of candidates across categories
    all_combinations = product(*candidate_options)

    results = []
    for combination in all_combinations:
        # Calculate the combined score for the current combination
        combined_score = sum(score for _, score in combination)
        results.append((combination, combined_score))

    # Sort results by score (descending)
    results.sort(key=lambda x: x[1], reverse=True)

    # Print results
    counter = 0
    print("\nCombined Scores for All Candidate Combinations:\n")
    for combination, score in results:
        if counter == 10:
            break
        assignment = " -> ".join(f"{category}: {candidate}" for category, (candidate, _) in zip(category_names, combination))
        print(f"{assignment} | Combined Score: {score:.3f}")
        counter = counter + 1

# Run the function
calculate_combined_scores(categories)