import pandas as pd
import ollama
import json
from tqdm import tqdm

# Constants
MODEL_NAME = "llama3.1:latest"  # Replace with your specific Ollama model name
FILE_PATH = "./data/non_sign_acc_actions.xlsx"
OUTPUT_PATH = "./data/non_sign_acc_actions_processed.xlsx"

def get_system_prompt():
    """
    Refined system prompt with explicit JSON constraints to satisfy Ollama's requirements.
    """
    return """
    You are a technical analyst specialized in software evolution. You will be provided with a Git diff (patch) for a Dependabot configuration file.
    
    ### TASK:
    1. Identify every 'option' being changed in the configuration.
    2. Map each change to the most appropriate 'label' from the list below.
    3. Extract the 'value' being introduced (the new configuration setting).
    
    ### MAPPING RULES:
    - Config file: Adopt Dependabot, Remove Dependabot, Rename file
    - schedule: Increase interval, Decrease interval, Add/set time, Add/set timezone
    - ignore: Ignore unstable dependencies/version, Unignore dependencies, Ignore ESM-related dependencies, Ignore dependency updates, Ignore major version upgrades, Ignore minor/patch version upgrades
    - Style: Fix typos
    - open-pull-request-limit: Increase limit, Decrease limit
    - groups: Group matching name dependencies, Group dev/prod dependencies, Ungroup prod/dev dependencies, Group all dependencies
    - versioning-strategy: Add/update versioning-strategy
    - reviewers: Add/update/remove reviewers
    - target-branch: Add/update/remove target-branch
    - commit-message: Add/update/remove commit-message
    - labels: Add/update/remove labels
    - package-ecosystem: Add/remove updates for npm
    - assignees: Add/update assignees
    - rebase-strategy: Add rebase-strategy
    - Others: Others

    ### OUTPUT FORMAT:
    CRITICAL: You must respond exclusively in valid JSON format. Do not include any conversational text, explanations, or Markdown formatting outside of the JSON block.
    
    Return a JSON list of objects. If a diff contains multiple changes, include an object for each change.
    Format: [{"option": "string", "label": "string"}]

    ### EXAMPLE:
    Patch: 
    -    open-pull-requests-limit: 5
    +    open-pull-requests-limit: 20
    Output: [{"option": "open-pull-requests-limit", "label": "Increase limit"}]
    """

def process_diff(diff_text):
    """
    Sends the diff to Ollama and extracts multiple findings if present.
    """
    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {'role': 'system', 'content': get_system_prompt()},
                {'role': 'user', 'content': f"Analyze this patch:\n{diff_text}"}
            ],
            format='json',
            options={
                'num_predict': 32000,  # This is the "max_tokens" equivalent
                'temperature': 0.1   # Keep it low for consistent extraction
            }
        )
        
        results = json.loads(response['message']['content'])
        
        if isinstance(results, list):
            # Join multiple findings with a semicolon for flat Excel storage
            options = "; ".join([str(r.get('option', '')) for r in results])
            labels = "; ".join([str(r.get('label', '')) for r in results])
            values = "; ".join([str(r.get('value', '')) for r in results])
            return options, labels, values
        return "", "", ""
    except Exception as e:
        return "error", "error", str(e)

def main():
    # Load the dataset
    df = pd.read_excel(FILE_PATH)
    
    if 'diff' not in df.columns:
        print("Error: 'diff' column not found.")
        return

    extracted_options = []
    extracted_labels = []
    extracted_values = []

    print(f"Starting extraction on {len(df)} rows...")

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        opt, lab, val = process_diff(row['diff'])
        extracted_options.append(opt)
        extracted_labels.append(lab)
        extracted_values.append(val)

    # Adding the new columns
    df['option'] = extracted_options
    df['label'] = extracted_labels
    df['new_value'] = extracted_values

    # Save results
    df.to_excel(OUTPUT_PATH, index=False)
    print(f"Successfully saved results to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()