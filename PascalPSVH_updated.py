import os
import scipy.io as sio
import numpy as np
import json
import numpy as np
import scipy.io as sio
from scipy.optimize import minimize_scalar
import cv2
import trimesh # 'pip install trimesh' is common in Colab
import json
import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

ROOT_PATH = '/Users/nehirgulcecesmeci/Downloads/PASCAL3D+_release1.1'
METADATA_PATH = '/Users/nehirgulcecesmeci/Desktop/pascal_metadata.json'
SAVE_DIR = '/Users/nehirgulcecesmeci/Desktop/ibcc'
MASK_SAVE_DIR = os.path.join(SAVE_DIR, 'masks')
VOXEL_SAVE_DIR = os.path.join(SAVE_DIR, 'voxels')
os.makedirs(MASK_SAVE_DIR, exist_ok=True)
os.makedirs(VOXEL_SAVE_DIR, exist_ok=True)

def build_pascal_metadata(root_dir, categories):
    print("🛠️ Building PASCAL 3D+ metadata...")
    metadata = {cat: [] for cat in categories}

    # Base path for annotations
    anno_root = os.path.join(root_dir, 'Annotations')

    if not os.path.exists(anno_root):
        print(f"❌ Error: {anno_root} not found!")
        return

    for cat in categories:
        # We look for folders containing the category name (e.g., 'chair_imagenet')
        subsets = [f'{cat}_imagenet', f'{cat}_pascal']

        for subset in subsets:
            subset_path = os.path.join(anno_root, subset)
            if not os.path.exists(subset_path):
                continue

            print(f"🔍 Processing {subset}...")

            # Use os.walk to find ALL .mat files even in sub-directories
            for root, dirs, files in os.walk(subset_path):
                for file in files:
                    if file.lower().endswith('.mat'):
                        file_path = os.path.join(root, file)

                        try:
                            # Load .mat
                            mat = sio.loadmat(
                                file_path,
                                struct_as_record=False,
                                squeeze_me=True
                            )
                            record = mat['record']

                            # Skip files with more than one object
                            if np.size(record.objects) > 1:
                                """  print(f"⚠️ Skipping {file}: multiple objects found")
                                print("Number of objects:", np.size(record.objects)) """
                                continue

                            # At this point: exactly 1 object
                            obj = record.objects
                            # process obj here

                            try:
                                # Check if 'class_name' exists; if not, check for 'class'
                                # Some PASCAL versions use obj.class instead of obj.class_name
                                cls_name = getattr(obj, 'class_name', getattr(obj, 'class', None))

                                """  if(subset.find("pascal") != -1):
                                print("cls_name:", cls_name)
                                print("occluded:", obj.occluded) #1 for occluded, 0 for not occluded
                                print("viewpoint:", obj.viewpoint)
                                """

                                if cls_name and str(cls_name).lower() == cat.lower():
                                    # Ensure viewpoint exists before accessing it
                                    if not hasattr(obj, 'viewpoint') or obj.viewpoint == [] or obj.occluded == 1 :
                                        continue  
                                    get_6d_pose_vector(azimuth=float(obj.viewpoint.azimuth), elevation=float(obj.viewpoint.elevation), theta=float(obj.viewpoint.theta), distance= float(obj.viewpoint.distance) )                                        
                                    entry = {
                                        "image_path": os.path.join(root_dir,'Images', subset, getattr(record, 'filename', file.replace('.mat', '.JPEG'))),
                                        "anno_path": file_path,
                                        "cad_index": int(getattr(obj, 'cad_index', 0)),
                                        "azimuth": float(obj.viewpoint.azimuth),
                                        "elevation": float(obj.viewpoint.elevation),
                                        "theta": float(obj.viewpoint.theta),
                                        "distance": float(obj.viewpoint.distance)
                                    }
                                    metadata[cat].append(entry)
                            except AttributeError as e:
                                # This will catch cases where viewpoint or other sub-attributes are missing
                                print(f"Skipping object in {file}: Missing attributes ({e})")
                                continue
                        except Exception as e:
                            print(f"⚠️ Error parsing {file}: {e}")

    # Save results
    with open(METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=4)

    for cat in categories:
        print(f"✅ {cat}: Found {len(metadata[cat])} objects")
    print(f"\nSUCCESS: Metadata saved to {METADATA_PATH}")

# Run the updated function
build_pascal_metadata(ROOT_PATH, ['chair', 'car', 'aeroplane', 'sofa']) #['chair', 'car', 'aeroplane', 'sofa']

def voxelize_cad(off_path, grid_size=32):
    """Converts a 3D .off mesh into a 32x32x32 binary voxel grid."""
    mesh = trimesh.load(off_path)
    # Center the mesh at the origin and normalize scale
    mesh.apply_translation(-mesh.centroid)
    
    # Create the voxel grid
    voxels = mesh.voxelized(pitch=mesh.extents.max()/grid_size).matrix
    
    # Ensure it is exactly the shape the network expects
    # Pad or crop to (32, 32, 32)
    return voxels

def save_pseudo_gt(save_dir, img_id, mask, d_corrected):
    """Saves the generated mask as an image and prepares the metadata entry."""
    mask_path = os.path.join(save_dir, f"{img_id}.png")
    cv2.imwrite(mask_path, mask)
    return d_corrected, mask_path

def get_2d_landmarks_with_names(anno_path):
    """Returns a dictionary of visible landmarks: {name: [x, y]}"""
    try:
        mat = sio.loadmat(anno_path, struct_as_record=False, squeeze_me=True)
        record = mat.record if hasattr(mat, 'record') else mat.get('record')
        obj_data = record.objects
        obj = obj_data[0] if isinstance(obj_data, (list, np.ndarray)) else obj_data

        landmarks = {}
        if hasattr(obj, 'anchors'):
            anchors = obj.anchors
            if not isinstance(anchors, (list, np.ndarray)):
                anchors = [anchors]
            
            for a in anchors:
                if getattr(a, 'status', 0) == 1:
                    name = getattr(a, 'name', None)
                    if name and hasattr(a, 'location'):
                        landmarks[name] = [a.location.x, a.location.y]
        return landmarks
    except:
        return {}

# ============================================================
# OFF LOADER (VERTICES + FACES)
# ============================================================
def load_off(path):
    with open(path, 'r') as f:
        lines = f.readlines()

    assert lines[0].strip() == 'OFF'
    n_verts, n_faces, _ = map(int, lines[1].split())

    verts = np.array([
        list(map(float, lines[i + 2].split()))
        for i in range(n_verts)
    ], dtype=np.float32)

    faces = np.array([
        list(map(int, lines[i + 2 + n_verts].split()[1:4]))
        for i in range(n_faces)
    ], dtype=np.int32)

    return verts, faces

# ============================================================
# NORMALIZE CAD MODEL SCALE
# ============================================================
def normalize_vertices_to_psvh_scale(vertices):
    """Aligns scaling logic with the PSVH radius-0.5 requirement."""
    center = vertices.mean(axis=0)
    verts = vertices - center
    max_dist = np.max(np.linalg.norm(verts, axis=1))
    # PSVH scale: radius = 0.5
    scale_factor = 0.5 / max_dist
    return verts * scale_factor

# ============================================================
# PROJECT 3D → 2D (DEPTH OPTIONAL)
# ============================================================
def project_3d_to_2d(
    model_pts,
    azimuth,
    elevation,
    theta,
    d,
    f=2000,
    img_size=224,
    return_depth=False
):
    a, e, t = np.radians(azimuth), np.radians(elevation), np.radians(theta)

    Rz = np.array([[np.cos(a), -np.sin(a), 0],
                   [np.sin(a),  np.cos(a), 0],
                   [0,          0,         1]])
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(e), -np.sin(e)],
                   [0, np.sin(e),  np.cos(e)]])
    Rt = np.array([[np.cos(t), -np.sin(t), 0],
                   [np.sin(t),  np.cos(t), 0],
                   [0,          0,         1]])

    R = Rt @ Rx @ Rz
    pts = (R @ model_pts.T).T

    z = pts[:, 2] + d
    proj = f * pts[:, :2] / z[:, None]

    proj += img_size / 2

    if return_depth:
        return proj, z
    return proj

# ============================================================
# AUTO-FIT 2D PROJECTION INTO IMAGE
# ============================================================
def fit_projection_to_image(proj, img_size=224, margin=10):
    min_xy = proj.min(axis=0)
    max_xy = proj.max(axis=0)

    size = max_xy - min_xy
    scale = (img_size - 2 * margin) / max(size)

    proj = (proj - min_xy) * scale + margin
    return proj

# ============================================================
# 3D ALIGNMENT & VOXELIZATION (RADIUS-0.5 REQUIREMENT)
# ============================================================
def align_to_canonical(off_path, grid_size=32):
    """
    Replaces old voxelize_cad.
    Centers mesh, scales to radius-0.5, and voxelizes to 32x32x32.
    """
    mesh = trimesh.load(off_path)
    
    # 1. Center the mesh at the origin
    mesh.apply_translation(-mesh.centroid)
    
    # 2. Normalize scale: fit in a radius-0.5 sphere
    # (Distance from center to furthest vertex = 0.5)
    max_dist = np.max(np.linalg.norm(mesh.vertices, axis=1))
    scale_factor = 0.5 / max_dist
    mesh.apply_scale(scale_factor)
    
    # 3. Voxelize
    # We use pitch=1.0/grid_size so the unit cube (1.0) maps to 32 voxels
    voxels_obj = mesh.voxelized(pitch=1.0/grid_size)
    voxels = voxels_obj.matrix
    
    # 4. Ensure exact (32, 32, 32) shape via padding/cropping
    final_voxels = np.zeros((grid_size, grid_size, grid_size), dtype=np.uint8)
    v_shape = voxels.shape
    
    sz = [min(grid_size, s) for s in v_shape]
    start = [(grid_size - s) // 2 for s in sz]
    v_start = [(s - min_s) // 2 for s, min_s in zip(v_shape, sz)]
    
    final_voxels[start[0]:start[0]+sz[0], 
                 start[1]:start[1]+sz[1], 
                 start[2]:start[2]+sz[2]] = voxels[v_start[0]:v_start[0]+sz[0],
                                                   v_start[1]:v_start[1]+sz[1],
                                                   v_start[2]:v_start[2]+sz[2]]
    return final_voxels

def visualize_voxels_with_axes(voxels, title="Voxel Check"):
    """Visualizes 3D voxels and indicates the Z-axis."""
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot voxels
    ax.voxels(voxels, edgecolor='k', alpha=0.7)
    
    # Draw Z-axis (Red Arrow) to check orientation
    # In canonical space, Z is often the 'depth' or 'forward' axis
    ax.quiver(16, 16, 16, 0, 0, 15, color='red', lw=3, label='Positive Z (Forward)')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    plt.legend()
    plt.show()
def verify_voxels(voxel_path):
    voxels = np.load(voxel_path)
    occupied_indices = np.argwhere(voxels > 0.5)
    
    if len(occupied_indices) == 0:
        return "Empty voxel grid!"

    # 1. Position Check: Center of occupancy
    center = np.mean(occupied_indices, axis=0)
    
    # 2. Scale Check: Maximum distance from center
    # Grid is 32x32x32, so center should be ~15.5. 
    # Max radius should be ~16 voxels (0.5 of the 32-unit cube)
    relative_coords = occupied_indices - np.array([15.5, 15.5, 15.5])
    max_radius = np.max(np.linalg.norm(relative_coords, axis=1))
    
    return {
        "center_offset": np.linalg.norm(center - 15.5),
        "max_radius_voxels": max_radius,
        "is_scaled_correctly": 14 < max_radius <= 16 # roughly 0.45-0.5 radius
    }

# ============================================================
# CALCULATING 6D VECTOR FROM AZIMUTH ELEVATION THETA DISTANCE, THIS IS REQUIRED FOR FINE TUNING
# ============================================================
def get_6d_pose_vector(azimuth, elevation, theta, distance, img_size=224, mask=None):
    # 1. Normalize Euler Angles to [0, 1] range as required by the paper
    # Based on PASCAL 3D+ defaults: azimuth (0-360), elevation (-90-90), theta (0-360)
    theta1 = (azimuth % 360) / 360.0
    theta2 = (elevation + 90) / 180.0
    theta3 = (theta % 360) / 360.0
    
    # 2. Calculate tu and tv (translation to centralize object on image plane)
    # Using the center of the generated mask
    img_center = img_size / 2.0
    if mask is not None and np.any(mask > 0):
        y_coords, x_coords = np.where(mask > 0)
        tu = np.mean(x_coords) - img_center
        tv = np.mean(y_coords) - img_center
    else:
        tu, tv = 0.0, 0.0 # Default if no mask is found
    
    # 3. Final 6D vector p = [theta1, theta2, theta3, tu, tv, tZ]
    p = [float(theta1), float(theta2), float(theta3), float(tu), float(tv), float(distance)]
    return p
# ============================================================
# FAST SILHOUETTE GENERATION (NO PYTHON PIXEL LOOPS)
# ============================================================
def generate_cad_mask(
    vertices,
    azimuth,
    elevation,
    theta,
    d_opt,
    cad_faces,
    img_size=224,
    smooth=True
):
    # Normalize model scale (CRITICAL)
    vertices = normalize_vertices_to_psvh_scale(vertices)

    # Project
    proj = project_3d_to_2d(
        vertices, azimuth, elevation, theta, d_opt, f=2000,
        img_size=img_size, return_depth=False
    )

    # 3. IMPORTANT: The PSVH paper DOES NOT 'fit to image' for masks.
    # The mask's position is determined by the pose (azimuth, elev, tu, tv, d).
    # If you use fit_projection_to_image, you are overwriting the 'tu' and 'tv' 
    # the network is trying to learn. 
    # REMOVE: proj = fit_projection_to_image(proj, img_size)

    proj = np.round(proj).astype(np.int32)

    mask = np.zeros((img_size, img_size), dtype=np.uint8)

    # FAST: fill triangles directly with OpenCV
    for face in cad_faces:
        tri = proj[face]
        if cv2.contourArea(tri) > 1:
            cv2.fillConvexPoly(mask, tri, 255)

    if smooth:
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    return mask

def optimize_distance(image_landmarks, model_3d_pts, azimuth, elevation, theta):
    """Finds the 'd' that minimizes the distance between projected and real points."""
    
    def objective(d):
        proj = project_3d_to_2d(model_3d_pts, azimuth, elevation, theta, d)
        # Calculate Mean Squared Error (MSE) between projected and real 2D points
        error = np.mean(np.linalg.norm(proj - image_landmarks, axis=1))
        return error

    # Search for optimal distance between 1.0 and 20.0 (typical range)
    res = minimize_scalar(objective, bounds=(1.5, 15.0), method='bounded')
    return res.x

# ============================================================
# MAIN PROCESSING LOOP
# ============================================================
def get_ready_for_training(visualize_first_n=3):
    landmark_fields = ['back_upper_left', 'back_upper_right', 'seat_upper_left', 'seat_upper_right',
                       'seat_lower_left', 'seat_lower_right', 'leg_upper_left', 'leg_upper_right',
                       'leg_lower_left', 'leg_lower_right']

    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)

    for cat, images in metadata.items():
        print(f"\n🚀 Processing category: {cat}")
        master_mat_path = os.path.join(ROOT_PATH, 'CAD', f"{cat}.mat")
        if not os.path.exists(master_mat_path): continue
        
        master_data = sio.loadmat(master_mat_path, struct_as_record=False, squeeze_me=True)
        cad_models = master_data.get(cat, master_data.get('model'))
        if not isinstance(cad_models, (list, np.ndarray)): cad_models = [cad_models]

        voxel_cache = {} 
        for count, entry in enumerate(images):
            try:
                img_id = os.path.basename(entry['image_path']).split('.')[0]
                cad_idx = entry['cad_index']
                current_model = cad_models[cad_idx-1]
                cad_path_off = os.path.join(ROOT_PATH, 'CAD', cat, f"{cad_idx:02d}.off")

                # 1. Optimize distance
                image_landmarks_dict = get_2d_landmarks_with_names(entry['anno_path'])
                aligned_2d, aligned_3d = [], []
                for field in landmark_fields:
                    if field in image_landmarks_dict and hasattr(current_model, field):
                        val_3d = getattr(current_model, field)
                        if isinstance(val_3d, np.ndarray) and val_3d.size == 3:
                            aligned_2d.append(image_landmarks_dict[field]); aligned_3d.append(val_3d)

                new_d = entry['distance']
                if len(aligned_2d) >= 3:
                    new_d = optimize_distance(np.array(aligned_2d), np.array(aligned_3d), 
                                              entry['azimuth'], entry['elevation'], entry['theta'])

                # 2. Generate Mask
                cad_vertices, cad_faces = load_off(cad_path_off)
                mask = generate_cad_mask(cad_vertices, entry['azimuth'], entry['elevation'], entry['theta'], new_d, cad_faces)
                _, mask_path = save_pseudo_gt(MASK_SAVE_DIR, img_id, mask, new_d)

                # 3. Canonical Voxelization (Radius 0.5)
                voxel_key = f"{cat}_{cad_idx:02d}"
                voxel_file = os.path.join(VOXEL_SAVE_DIR, f"{voxel_key}.npy")
                
                if voxel_key not in voxel_cache:
                    # UPDATED: Using align_to_canonical instead of voxelize_cad
                    voxels = align_to_canonical(cad_path_off)
                    np.save(voxel_file, voxels)
                    voxel_cache[voxel_key] = voxel_file
                    
                    stats = verify_voxels(voxel_file)
                    print(f"📊 Voxel Stats for {voxel_key}: {stats}")
                    
                    if not stats["is_psvh_compliant"]:
                        print(f"⚠️ WARNING: {voxel_key} might be misaligned or incorrectly scaled!")
                    
                    if count < visualize_first_n:
                        visualize_voxels_with_axes(voxels, title=f"Check: {voxel_key}\nRadius: {stats['max_radius_voxels']}")

                # 4. Final Metadata Update
                entry['pose_6d'] = get_6d_pose_vector(entry['azimuth'], entry['elevation'], entry['theta'], new_d, mask=mask)
                entry['corrected_d'] = float(new_d)
                entry['mask_path'] = mask_path
                entry['voxel_gt_path'] = voxel_file

                if count % 50 == 0:
                    print(f"  ✅ {cat}: {count}/{len(images)} processed")

            except Exception as e:
                print(f"❌ FAILED on {entry['image_path']}: {e}")
                raise

    with open(os.path.join(SAVE_DIR, 'pascal_metadata_training.json'), 'w') as f:
        json.dump(metadata, f, indent=4)
    print("🎉 Done!")
get_ready_for_training(visualize_first_n=5)
