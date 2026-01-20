import os
import json

# Define the path to your main ShapeNetRendering folder
SHAPENET_RENDERING_DIR = 'ShapeNetRendering'

# Define the categories 
TARGET_CATEGORIES = [
    "04256520", "02691156", "03636649", "04401088", 
    "04530566", "03691459", "03001627", "02933112",
    "04379243", "03211117", "02958343", "02828884", 
    "04090263"
]

def generate_sorted_model_list(rendering_dir, target_categories):
    """Generates a dictionary of {Category ID: [Sorted Model ID List]}."""
    
    full_model_list = {}
    print(f"Scanning directory: {rendering_dir}")

    for cat_id in target_categories:
        cat_path = os.path.join(rendering_dir, cat_id)
        
        if not os.path.isdir(cat_path):
            print(f"  Skipping {cat_id}: Directory not found.")
            continue
            
        # Collect all Model IDs (subdirectories that are not 'rendering')
        model_ids = [
            d for d in os.listdir(cat_path) 
            if os.path.isdir(os.path.join(cat_path, d))
        ]
        
        # Sort the Model IDs lexicographically (alphabetically)
        # This establishes the canonical order for the 80/20 split.
        model_ids.sort()
        
        full_model_list[cat_id] = model_ids
        print(f"  Found {len(model_ids)} unique models for category {cat_id}.")

    return full_model_list

# Generate the list based on local files
full_data = generate_sorted_model_list(SHAPENET_RENDERING_DIR, TARGET_CATEGORIES)

# Ensure the output directory exists
output_dir = './experiments/dataset'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save the generated list to a file that your evaluation script can load
output_path = os.path.join(output_dir, 'derived_full_model_list.json')

with open(output_path, 'w') as f:
    json.dump(full_data, f, indent=4)

print(f"\n✅ Successfully generated the Model ID list to: {output_path}")
