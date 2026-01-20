import numpy as np
import tensorflow as tf
import os
import time
import json
from PIL import Image
import random 
import binvox_rw 
from run_case import syn_model, real_model 
from voxel import voxel2obj

# --- Configuration and Constants ---
voxel_size = 32
img_h = 128
img_w = 128
threshold = 0.4 
TARGET_BASE_DIR = '../shapenet_data' 

# Path to the derived, alphabetically sorted list of Model IDs
FULL_MODEL_ID_JSON = '../experiments/dataset/derived_full_model_list.json' 

# Categories mapping
DIC = {"04256520": "sofa", "02691156": "airplane", "03636649": "lamp", "04401088": "telephone",
       "04530566": "vessel", "03691459": "loudspeaker", "03001627": "chair", "02933112": "cabinet",
       "04379243": "table", "03211117": "display", "02958343": "car", "02828884": "bench", "04090263": "rifle"}

# Define the testing portion as in the paper (last 20% of the sorted list) 
TEST_PORTION = [0.8, 1] 

# This is the most likely correct normalization for the pre-trained model?
PIXEL_MEAN_NEG1_TO_1 = np.array([0.5, 0.5, 0.5], dtype=np.float32)
PIXEL_STD_NEG1_TO_1 = np.array([0.5, 0.5, 0.5], dtype=np.float32)
# ----------------------------------------------------------------

# --- Utility Functions ---

def load_gt_voxel(filepath):
    """Loads a binary voxel grid from a .binvox file."""
    try:
        with open(filepath, 'rb') as f:
            model = binvox_rw.read_as_3d_array(f)
            return model.data
    except Exception as e:
        return None

def calculate_iou(pred, gt):
    """Calculates Intersection over Union (IoU) between prediction and ground truth."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    intersection = np.sum(np.logical_and(pred, gt))
    union = np.sum(np.logical_or(pred, gt))
    return intersection / union if union > 0 else 0.0

def load_full_model_ids(json_path=FULL_MODEL_ID_JSON):
    """Loads ALL ordered model IDs (train + test) from the derived JSON file."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"FATAL: Full Model ID List not found at {json_path}.")
        print("Please ensure the 'derived_full_model_list.json' file is generated and placed correctly.")
        return None

def get_test_model_ids(category_name, all_data, test_portion):
    """Filters the model IDs based on the TEST_PORTION [0.8, 1]."""
    
    cat_id = next(key for key, value in DIC.items() if value == category_name)
    
    if cat_id not in all_data:
        print(f"Warning: Category ID {cat_id} not found in the loaded model list.")
        return set()
    
    model_list = all_data[cat_id]
    
    if not isinstance(model_list, list):
         print(f"FATAL: Data for {cat_id} is not a list of Model IDs. Check your derived JSON file structure.")
         return set()

    total_models = len(model_list)
    print("total number of models:", total_models)
    start_index = int(test_portion[0] * total_models)
    end_index = int(test_portion[1] * total_models)
    
    # Extract the test set using slicing on the pre-sorted list
    test_ids = set(model_list[start_index:end_index])
    
    return test_ids

# --- Main Evaluation Function ---

def evaluate_shapenet(model_type='syn', category_name='chair'):
    """
    Evaluates the PSVH model on the designated TEST split of the ShapeNet data.
    """
    
    # --- Mesh Saving Configuration ---
    SAVE_MESHES = True 
    SAVE_SAMPLE_PERCENTAGE = 0.05  # Save 5% of all test models
    OUTPUT_MESH_DIR = os.path.join('../output/reconstructions', category_name)
    # NOTE: This directory is IGNORED by the voxel2obj function, which writes to 
    # the hardcoded 'evaluation_output' folder. We keep this path for naming purposes.
    if SAVE_MESHES and not os.path.exists(OUTPUT_MESH_DIR):
        os.makedirs(OUTPUT_MESH_DIR)
    # ---------------------------------

    # --- 1. Setup ---
    cat_dir = os.path.join(TARGET_BASE_DIR, category_name)
    weight_path = os.path.join(f'../{model_type}_model', 'model.cptk')
    
    if not os.path.exists(cat_dir):
        print(f"FATAL: Organized data directory not found: {cat_dir}")
        return

    all_split_data = load_full_model_ids() 
    if all_split_data is None:
        return 

    test_model_ids = get_test_model_ids(category_name, all_split_data, TEST_PORTION)
    #print("test model ids",test_model_ids)
    
    if not test_model_ids:
        print(f"FATAL: Could not derive any test samples for {category_name}. Exiting.")
        return

    # Filter files based on the derived test model IDs
    files_to_process = []
    for filename in os.listdir(cat_dir):
        if filename.endswith('.png'):
            model_id = filename.rsplit('_', 1)[0]
            """  print("model id:", model_id)
            print("Example filenames:", os.listdir(cat_dir)[:10])
            print("Example test model ids:", list(test_model_ids)[:10])
            """
            if model_id in test_model_ids:
                #print("added file:", filename)
                files_to_process.append(filename)

    if not files_to_process:
        print(f"FATAL: Found {len(test_model_ids)} test models, but no corresponding images in {cat_dir}.")
        return

    # --- 2. Build Model and Session ---
    if model_type == 'syn':
        before, after, img_input = syn_model()
    else:
        before, after, img_input = real_model() 
    
    params = tf.trainable_variables()
    saver = tf.train.Saver(var_list=params)
    
    total_iou = 0.0
    valid_samples = 0
    
    print(f"\n--- Starting Evaluation for {category_name.upper()} ({len(files_to_process)} test samples) ---")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(f"Loading checkpoint from: {weight_path}")
        saver.restore(sess, weight_path) 
        print("Checkpoint loaded successfully.")

        start_time = time.time()
        
        # --- 3. Run Inference and Calculate IoU ---
        for filepath_name in files_to_process:
            model_view_prefix = filepath_name[:-4] 
            model_id = model_view_prefix.rsplit('_', 1)[0]
            
            # Construct paths
            voxel_gt_path = os.path.join(cat_dir, f"{model_id}_voxel.binvox")
            """ print("Processing filepath_name:", filepath_name)
            print("voxel_gt_path:", voxel_gt_path) """
            img_filepath = os.path.join(cat_dir, filepath_name)

            #print("img_filepath:", img_filepath)
            
            # Load and preprocess image
            img = Image.open(img_filepath).convert('RGB').resize((img_h, img_w))
            img = np.array(img).astype(np.float32) / 255.  # Step 1: Normalize to [0, 1]
            
            img = (img - PIXEL_MEAN_NEG1_TO_1) / PIXEL_STD_NEG1_TO_1 
            
            img_feed = img.reshape([1, img_h, img_w, 3])
            
            # Load GT Voxel
            v_gt = load_gt_voxel(voxel_gt_path)

            
            if v_gt is None:
                print(f"  ⚠️ Warning: Could not load GT voxel for {model_id}. Skipping sample.")
                continue 
            
            # Run inference
            v_after = sess.run(after, feed_dict={img_input: img_feed})
            
            # Convert prediction to binary using the threshold (0.4)
            v_pred_binary = v_after.squeeze() > threshold
            
            # Calculate IoU
            current_iou = calculate_iou(v_pred_binary, v_gt)
            total_iou += current_iou
            valid_samples += 1
            
            if SAVE_MESHES:
                view_number = filepath_name.rsplit('_', 1)[1].split('.')[0]
                
                # Check for first view ('00') AND random chance (5%)
                if view_number == '00' and random.random() < SAVE_SAMPLE_PERCENTAGE:
                    mesh_output_path = os.path.join(OUTPUT_MESH_DIR, f"{model_id}_{view_number}.obj")
                    
                    # Call voxel2obj without the failing cube_size argument
                    # This function is assumed to write the file itself to 'evaluation_output'
                    voxel2obj(mesh_output_path, v_pred_binary) 
                    
                    # NOTE: Manual file writing is DELETED here as the voxel2obj 
                    # function is known to write the file to a hardcoded location.
                    print(f"  -> Triggered mesh save for {model_id}. Check 'evaluation_output'.")

            if valid_samples % 100 == 0:
                 print(f"  Processed {valid_samples} samples. Current avg IoU: {total_iou / valid_samples:.4f}")

        end_time = time.time()
        
        # --- 3. Final Results ---
        if valid_samples > 0:
            avg_iou = total_iou / valid_samples
            print("\n-------------------------------------------")
            print(f"✅ Evaluation Complete for {category_name.upper()}")
            print(f"  Total Samples Processed: {valid_samples}")
            print(f"  Time Taken: {end_time - start_time:.2f} seconds")
            print(f"  Mean IoU ({model_type.upper()} Model): {avg_iou:.4f}")
            print("-------------------------------------------")
        else:
            print("No valid samples were processed.")

if __name__ == '__main__':
    # Run the evaluation for the 'real' model on the 'chair' category by default
    evaluate_shapenet(model_type='syn', category_name='sofa')
