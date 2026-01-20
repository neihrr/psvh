import os
import shutil
import numpy as np

# --- Configuration ---

# The root directory where you extracted the ShapeNetVox32 data.
# ASSUMPTION: This directory contains the numerical category folders (e.g., '03001627').
SHAPENET_VOXEL_ROOT = 'ShapeNetVox32' 

# Root directory where you extracted the ShapeNetRendering data (contains numerical folders)
SHAPENET_RENDERING_ROOT = 'ShapeNetRendering' 

# The target directory where the organized data will be saved.
# This will create 'shapenet_data/chair', 'shapenet_data/table', etc.
TARGET_BASE_DIR = 'shapenet_data' #this is equivalent to './shapenet_data' it gives the relative path meanin that the folder will be created in the current working directory (CWD)

# Categories from the run_case.py script
CATES = ["04256520", "02691156", "03636649", "04401088",
         "04530566", "03691459", "03001627", "02933112",
         "04379243", "03211117", "02958343", "02828884", "04090263"] 

DIC = {"04256520": "sofa", "02691156": "airplane", "03636649": "lamp", "04401088": "telephone",
       "04530566": "vessel", "03691459": "loudspeaker", "03001627": "chair", "02933112": "cabinet",
       "04379243": "table", "03211117": "display", "02958343": "car", "02828884": "bench", "04090263": "rifle"}

def organize_shapenet_voxels():
    """
    Iterates through all specified ShapeNet categories, finds model.binvox files,
    renames them by model ID, and moves them to the target structured directory.
    """
    
    if not os.path.exists(TARGET_BASE_DIR):
        os.makedirs(TARGET_BASE_DIR)
        print(f"Created target directory: {TARGET_BASE_DIR}")

    total_files_processed = 0

    for cat_id in CATES:
        cat_name = DIC[cat_id]
        
        # 1. Define source and target paths for the current category
        source_cat_dir = os.path.join(SHAPENET_VOXEL_ROOT, cat_id)
        target_cat_dir = os.path.join(TARGET_BASE_DIR, cat_name)
        
        if not os.path.isdir(source_cat_dir):
            print(f"⚠️ Warning: Source directory not found for {cat_name} ({cat_id}). Skipping.")
            continue
            
        if not os.path.exists(target_cat_dir):
            os.makedirs(target_cat_dir)
            
        print(f"\nProcessing category: {cat_name} ({cat_id})")
        
        # 2. Iterate through all Model ID folders inside the category
        for model_id in os.listdir(source_cat_dir):
            model_source_dir = os.path.join(source_cat_dir, model_id)
            voxel_source_path = os.path.join(model_source_dir, 'model.binvox')
            
            # Check if the path is a directory and contains the voxel file
            if os.path.isdir(model_source_dir) and os.path.exists(voxel_source_path):
                # 3. Define the new file name (Model ID + suffix)
                new_file_name = f"{model_id}_voxel.binvox"
                voxel_target_path = os.path.join(target_cat_dir, new_file_name)
                
                # 4. Move and rename the file
                shutil.copy2(voxel_source_path, voxel_target_path)
                total_files_processed += 1
                
        print(f"✅ Finished {cat_name}. Processed {len(os.listdir(target_cat_dir))} files.")

def organize_shapenet_renderings():
    """
    Iterates through ShapeNet renderings, moves the PNG images, and extracts/saves 
    the 4 primary pose parameters (Azimuth, Elevation, Rotation, Distance) 
    from rendering_metadata.txt into separate .pose files.
    """
    
    if not os.path.exists(TARGET_BASE_DIR):
        print(f"Target directory {TARGET_BASE_DIR} not found. Please run voxel organization first.")
        return

    total_files_processed = 0

    for cat_id in CATES:
        cat_name = DIC[cat_id]
        
        # Define source and target paths
        source_cat_dir = os.path.join(SHAPENET_RENDERING_ROOT, cat_id)
        target_cat_dir = os.path.join(TARGET_BASE_DIR, cat_name)
        
        if not os.path.isdir(source_cat_dir):
            print(f"⚠️ Warning: Rendering source directory not found for {cat_name}. Skipping.")
            continue
        
        # Ensure the target category folder exists
        if not os.path.exists(target_cat_dir):
            os.makedirs(target_cat_dir)
            
        print(f"\nProcessing renderings for category: {cat_name} ({cat_id})")
        
        # Iterate through all Model ID folders
        for model_id in os.listdir(source_cat_dir):
            model_source_dir = os.path.join(source_cat_dir, model_id, 'rendering')
            metadata_path = os.path.join(model_source_dir, 'rendering_metadata.txt')
            
            if os.path.isdir(model_source_dir) and os.path.exists(metadata_path):
                
                # Load all metadata lines
                try:
                    metadata = np.loadtxt(metadata_path)
                except Exception as e:
                    print(f"Error loading metadata for {model_id}: {e}. Skipping.")
                    continue

                # We only care about the first 4 columns: Azimuth, Elevation, In-plane Rot, Distance
                pose_params = metadata[:, :4]
                
                # Iterate through all 24 views (assuming 00.png to 23.png)
                for i in range(24):
                    view_id = str(i).zfill(2)
                    png_file = f"{view_id}.png"
                    
                    # 1. Image Path Handling
                    image_source_path = os.path.join(model_source_dir, png_file)
                    image_target_name = f"{model_id}_{view_id}.png"
                    image_target_path = os.path.join(target_cat_dir, image_target_name)
                    
                    if os.path.exists(image_source_path):
                        shutil.copy2(image_source_path, image_target_path)
                        
                        # 2. Pose Data Handling
                        pose_target_name = f"{model_id}_{view_id}.pose"
                        pose_target_path = os.path.join(target_cat_dir, pose_target_name)
                        
                        # Save the 4 pose parameters (Azimuth, Elevation, Rotation, Distance) 
                        # This structure is needed for custom PSVH data loading.
                        np.savetxt(pose_target_path, pose_params[i], fmt='%.6f', delimiter=' ')
                        total_files_processed += 1
                        
            # else: skip models without rendering data or metadata
            
        print(f"✅ Finished {cat_name}. Processed {len([f for f in os.listdir(target_cat_dir) if f.endswith('.png')])} images and poses.")


organize_shapenet_voxels()
#organize_shapenet_renderings()