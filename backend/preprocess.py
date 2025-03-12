import pandas as pd  

# Define dataset paths  
datasets = {
    "aptitude": "/home/rouqh/Desktop/alvocate/data/clean_general_aptitude_dataset.csv",
    "logical_reasoning": "/home/rouqh/Desktop/alvocate/data/logical_reasoning_questions.csv",
    "cse": "/home/rouqh/Desktop/alvocate/data/cse_dataset.csv",
    "leetcode": "/home/rouqh/Desktop/alvocate/data/Leetcode_Questions.csv"
}

# Define expected column mappings
expected_columns_map = {
    "aptitude": {"question": "Question", "answer": "Answer"},
    "logical_reasoning": {"question": "Question", "answer": "Answer"},
    "cse": {"question": "Question", "answer": "Answer"},
    "leetcode": {"question": "Question", "answer": "Solution"}  # Assuming 'Solution' is the answer
}

# Function to load and process datasets
def load_and_process_dataset(name, filepath, delimiter=","):
    print(f"\nProcessing {name} dataset...")

    try:
        # Auto-detect if the dataset uses semicolons
        with open(filepath, "r", encoding="utf-8") as f:
            first_line = f.readline()
            if ";" in first_line and "," not in first_line:
                delimiter = ";"

        # Load dataset with correct delimiter
        df = pd.read_csv(filepath, delimiter=delimiter, encoding="utf-8", on_bad_lines="skip", quoting=3)
        print(f"Columns in {name} dataset:", df.columns)

        # Check if required columns exist
        if set(expected_columns_map[name].values()).issubset(df.columns):
            df = df[list(expected_columns_map[name].values())]  # Keep only relevant columns
            df.rename(columns={v: k for k, v in expected_columns_map[name].items()}, inplace=True)
        else:
            raise ValueError(f"Dataset '{name}' does not contain required columns: {list(expected_columns_map[name].values())}")

        print(f"Processed {name} dataset successfully.")
        return df

    except Exception as e:
        print(f"Error processing {name} dataset: {e}")
        return None

# Process each dataset
processed_datasets = {name: load_and_process_dataset(name, path) for name, path in datasets.items()}

# Save processed datasets
for name, df in processed_datasets.items():
    if df is not None:
        output_path = f"/home/rouqh/Desktop/alvocate/data/processed_{name}.csv"
        df.to_csv(output_path, index=False)
        print(f"Saved processed {name} dataset to {output_path}")
