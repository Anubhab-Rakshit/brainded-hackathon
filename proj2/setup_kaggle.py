import os
import json

def setup_kaggle_json():
    print("To download datasets from Kaggle, you need a 'kaggle.json' file.")
    print("1. Go to https://www.kaggle.com/settings")
    print("2. Click 'Create New Token' under the API section.")
    print("3. This will download a file named 'kaggle.json'.")
    
    kaggle_dir = os.path.expanduser("~/.kaggle")
    if not os.path.exists(kaggle_dir):
        os.makedirs(kaggle_dir)
        print(f"Created directory: {kaggle_dir}")
    
    target_path = os.path.join(kaggle_dir, "kaggle.json")
    
    if os.path.exists(target_path):
        print(f"Found existing credentials at: {target_path}")
        return
        
    print(f"\nPlease paste the content of your kaggle.json file below (or path to it):")
    user_input = input("Content or Path: ").strip()
    
    if user_input.endswith(".json") and os.path.exists(user_input):
        with open(user_input, 'r') as f:
            data = f.read()
    else:
        # Assume it's the json content string if it starts with {
        if not user_input.startswith("{"):
             # Maybe they just pasted the username/key raw?
             print("Invalid input. Please provide the JSON content or path to the file.")
             return
        data = user_input

    with open(target_path, 'w') as f:
        f.write(data)
    
    # Set permissions
    os.chmod(target_path, 0o600)
    print(f"Success! Credentials saved to {target_path}")

if __name__ == "__main__":
    setup_kaggle_json()
