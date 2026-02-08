import json
import os

def create_notebook(filename, cells):
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    with open(filename, 'w') as f:
        json.dump(notebook, f, indent=1)
    print(f"Created {filename}")

def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.split("\n")]
    }

def markdown_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")]
    }

# --- REELSENSE NOTEBOOK ---
reelsense_cells = [
    markdown_cell("# ReelSense: Advanced Hybrid Recommender System\n**Hackathon Submission**\n\nThis notebook demonstrates the end-to-end pipeline of ReelSense, including Data Loading, EDA, Model Training (SVD, Hybrid, NeuralCF, LightGCN), and Validated Evaluation."),
    
    markdown_cell("## 1. Setup and Initialization"),
    code_cell("!git clone https://github.com/Anubhab-Rakshit/brainded-hackathon.git\n%cd brainded-hackathon\n!pip install -r requirements.txt"),
    
    code_cell("import sys\nimport os\nimport pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom tqdm import tqdm\n\n# Add project to path\nsys.path.append(os.getcwd())\n\nfrom proj1.src.data import DataLoader\nfrom proj1.src.recommenders import *\nfrom proj1.src.evaluation import *\nfrom proj1.src.explainability import Explainer\n\n%matplotlib inline"),
    
    markdown_cell("## 2. Data Loading & Preprocessing\nWe use the MovieLens Small dataset."),
    code_cell("# Initialize Loader\nloader = DataLoader('ml-latest-small')\nloader.load_data()\nloader.preprocess()\n\n# Train-Test Split (Time-based for realism)\ntrain_df, test_df = loader.get_train_test_split(method='leave_last_n', n=1)\nprint(f\"Train Samples: {len(train_df)}\")\nprint(f\"Test Samples: {len(test_df)}\")\ntrain_df.head()"),
    
    markdown_cell("## 3. Exploratory Data Analysis (EDA)\nVisualizing the long-tail distribution and user activity."),
    code_cell("# 1. Long Tail Plot\nplt.figure(figsize=(10, 6))\nitem_counts = train_df['movieId'].value_counts()\nplt.plot(item_counts.values)\nplt.title('Long-Tail Distribution of Movie Ratings')\nplt.xlabel('Movie Rank')\nplt.ylabel('Number of Ratings')\nplt.yscale('log')\nplt.show()\n\n# 2. Ratings Distribution\nplt.figure(figsize=(8, 4))\nsns.countplot(x='rating', data=train_df)\nplt.title('Distribution of Ratings')\nplt.show()"),
    
    markdown_cell("## 4. Model Training & Evaluation\nWe train and compare multiple architectures: SVD, NeuralCF, and LightGCN."),
    code_cell("# Initialize Models\nmodels = {\n    \"Popularity\": PopularityRecommender(),\n    \"User-User CF\": CollaborativeRecommender(method='user_user', n_neighbors=50),\n    \"SVD (Matrix Factorization)\": SVDRecommender(n_components=20),\n    \"Neural CF (Deep Learning)\": NeuralCFRecommender(embedding_dim=32, n_epochs=5),\n    \"LightGCN (Graph NN)\": LightGCNRecommender(n_epochs=10)\n}\n\nresults = {}\nall_items = loader.movies['movieId'].unique()\n\nfor name, model in models.items():\n    print(f\"Training {name}...\")\n    model.fit(train_df)\n    metrics, _ = evaluate_model(model, train_df, test_df, all_items, k=10)\n    results[name] = metrics\n    print(f\"{name}: {metrics}\")"),
    
    markdown_cell("## 5. Final Results Calculation"),
    code_cell("results_df = pd.DataFrame(results).T\nresults_df = results_df[['Precision@K', 'Recall@K', 'NDCG@K', 'Coverage']]\n\n# Display Leaderboard\nresults_df.sort_values(by='NDCG@K', ascending=False)"),
    
    markdown_cell("## 6. Explainability Demo\nWhy did we recommend this?"),
    code_cell("# Explain a recommendation for a sample user\nexplainer = Explainer(loader, train_df)\nsample_user = test_df['userId'].iloc[0]\n\nrecs = models['SVD (Matrix Factorization)'].recommend(sample_user, n=3)\nprint(f\"User Profile: {explainer.user_profiles.get(sample_user, 'Unknown')}\")\n\nfor movie_id in recs:\n    title = loader.movies[loader.movies['movieId'] == movie_id]['title'].iloc[0]\n    explanation = explainer.explain(sample_user, movie_id)\n    print(f\"\\n🎥 Movie: {title}\")\n    print(f\"💡 Reason: {explanation}\")")
]

# --- COGNITIVE RADIOLOGY NOTEBOOK ---
cograd_cells = [
    markdown_cell("# Cognitive Radiology: AI Medical Report Generation\n**Hackathon Submission (Project 2)**\n\nThis notebook demonstrates the inference pipeline for our Transformer-based Radiology Reporter. It detects pathologies in Chest X-Rays and generates professional text reports."),
    
    markdown_cell("## 1. Environment Setup"),
    code_cell("!git clone https://github.com/Anubhab-Rakshit/brainded-hackathon.git\n%cd brainded-hackathon\n!pip install torch torchvision timm transformers pandas"),
    
    markdown_cell("## 2. Load Model & Tools"),
    code_cell("import torch\nimport matplotlib.pyplot as plt\nfrom PIL import Image\nfrom proj2.src.inference import load_model, generate_report\nfrom transformers import BertTokenizer\n\ndevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\nprint(f'Using Device: {device}')\n\n# Load Model (Ensure weights are present or downloaded)\n# Note: In Colab, you might need to upload model_final.pth to proj2/ directory manually if not in repo\ncheckpoint_path = 'proj2/model_final.pth'\n\nif not os.path.exists(checkpoint_path):\n    print(\"WARNING: Model file not found. Please upload 'model_final.pth' to proj2/ folder.\")\nelse:\n    model = load_model(checkpoint_path, device)\n    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')\n    print(\"Model Loaded Successfully\")"),
    
    markdown_cell("## 3. Training Pipeline (Demo)\nThis section demonstrates how we trained the model using MIMIC-CXR. \n**Note:** Running this requires the full 4TB dataset. We include a mock training loop here for verification."),
    code_cell("# Training Configuration\nimport torch.nn as nn\nimport torch.optim as optim\nfrom tqdm import tqdm\nfrom proj2.config import config\nfrom proj2.src.data.dataset import MockRadiologyDataset, get_transforms, get_tokenizer\nfrom proj2.src.model import CognitiveRadiologyModel\n\n# Initialize Mock Data for Demo\ntokenizer = get_tokenizer()\ntrain_dataset = MockRadiologyDataset(tokenizer=tokenizer, transforms=get_transforms(is_train=True), num_samples=10)\ntrain_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True)\n\n# Initialize Model\nmodel = CognitiveRadiologyModel(config).to(device)\noptimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)\ncriterion_cls = nn.BCEWithLogitsLoss()\ncriterion_gen = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)\n\n# Training Loop (1 Epoch Demo)\nprint(\"Starting Training Loop...\")\nmodel.train()\nfor batch in tqdm(train_loader):\n    images = batch['image'].to(device)\n    input_ids = batch['input_ids'].to(device)\n    labels = batch['labels'].to(device)\n    \n    optimizer.zero_grad()\n    \n    # Forward\n    decoder_input = input_ids[:, :-1]\n    decoder_target = input_ids[:, 1:]\n    outputs = model(images, input_ids=decoder_input, labels=labels)\n    \n    # Loss Calculation\n    loss_cls = criterion_cls(outputs['cls_logits'], labels)\n    loss_gen = criterion_gen(outputs['decoder_logits'].reshape(-1, tokenizer.vocab_size), decoder_target.reshape(-1))\n    loss = loss_cls + loss_gen\n    \n    # Backward\n    loss.backward()\n    optimizer.step()\n    \nprint(\"Training Demo Completed.\")"),

    markdown_cell("## 4. Inference Demo\nWe test the TRAINED model on a Pneumonia case."),
    code_cell("# Download Test Image (Pneumonia)\n!curl -o demo_pneumonia.jpg https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/000009-5.jpg\n\n# Display Image\nimg = Image.open('demo_pneumonia.jpg')\nplt.figure(figsize=(6,6))\nplt.imshow(img, cmap='gray')\nplt.title(\"Input Chest X-Ray\")\nplt.axis('off')\nplt.show()\n\n# Run Inference\nprint(\"Generating Report...\")\n# Run via command line to use the full script logic/formatting\n!python3 -m proj2.src.inference --image demo_pneumonia.jpg --checkpoint proj2/model_final.pth")
]

create_notebook("notebooks/ReelSense_Submission.ipynb", reelsense_cells)
create_notebook("notebooks/CognitiveRadiology_Demo.ipynb", cograd_cells)
