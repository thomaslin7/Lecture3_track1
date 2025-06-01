import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon
from scipy.spatial.transform import Rotation as R
import random

class Scene3DVisualizer:
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.objects = []   # List of objects drawn in the scene
        
    def draw_grid_background(self, img, grid_size=50, color=[80, 80, 80]):
        """Draw grid lines in the background"""
        height, width = img.shape[:2]
        
        # Draw vertical lines
        for x in range(0, width, grid_size):
            cv2.line(img, (x, 0), (x, height), color, 1)
        
        # Draw horizontal lines
        for y in range(0, height, grid_size):
            cv2.line(img, (0, y), (width, y), color, 1)
        
    def create_scene(self):
        """Create a scene with different geometric objects and depth values"""
        # Create RGB image
        rgb_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        depth_image = np.zeros((self.height, self.width), dtype=np.float32)
        
        # Add grid background
        self.draw_grid_background(rgb_image)    # NumPy arrays (like rgb_image) are mutable objects
        
        # Object definitions with (x, y, size, depth, color, shape, fill)
        objects_data = [
            # Blue square (hollow)
            {'center': (200, 150), 'size': 70, 'depth': 0.2, 'color': [220, 10, 20], 'shape': 'square', 'fill': False},
            # Pink square (hollow)
            {'center': (500, 450), 'size': 85, 'depth': 0.4, 'color': [255, 0, 255], 'shape': 'square', 'fill': False},
            # Cyan triangle (solid)
            {'center': (300, 400), 'size': 110, 'depth': 0.6, 'color': [255, 255, 0], 'shape': 'triangle', 'fill': True},
            # Purple triangle (solid)
            {'center': (600, 180), 'size': 95, 'depth': 0.8, 'color': [200, 0, 200], 'shape': 'triangle', 'fill': True},
        ]
        
        for obj in objects_data:
            self.draw_object(rgb_image, depth_image, obj)
            self.objects.append(obj)

        # For debugging
        cv2.imshow("Original RGB Image", rgb_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imshow("Original Depth Image", depth_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return rgb_image, depth_image
    
    def draw_object(self, rgb_img, depth_img, obj):
        """Draw individual objects on RGB and depth images"""
        center = obj['center']
        size = obj['size']
        depth = obj['depth']
        color = obj['color']
        shape = obj['shape']
        fill = obj['fill']
        
        if shape == 'square':
            self.draw_square(rgb_img, depth_img, center, size, depth, color, fill)
        elif shape == 'triangle':
            self.draw_triangle(rgb_img, depth_img, center, size, depth, color, fill)
    
    def draw_square(self, rgb_img, depth_img, center, size, depth, color, fill):
        """Draw square on images"""
        x, y = center
        half_size = size // 2 # floor division for integer size
        
        if fill:
            # Draw filled rgb
            cv2.rectangle(rgb_img, (x-half_size, y-half_size), 
                         (x+half_size, y+half_size), color, -1) # -1 indicates filled
            # Draw filled depth
            cv2.rectangle(depth_img, (x-half_size, y-half_size), 
                         (x+half_size, y+half_size), depth, -1)
        else:
            # Draw hollow rgb
            cv2.rectangle(rgb_img, (x-half_size, y-half_size), 
                         (x+half_size, y+half_size), color, 10)
            # Draw hollow depth
            cv2.rectangle(depth_img, (x-half_size, y-half_size), 
                         (x+half_size, y+half_size), depth, 10)
    
    def draw_triangle(self, rgb_img, depth_img, center, size, depth, color, fill):
        """Draw triangle on images"""
        x, y = center
        height = int(size * 0.866)  # equilateral triangle height
        
        pts = np.array([
            [x, y - height//2], # Top vertex
            [x - size//2, y + height//2],   # Bottom left vertex
            [x + size//2, y + height//2]    # Bottom right vertex
        ], np.int32)
        
        if fill:
            cv2.fillPoly(rgb_img, [pts], color)
            cv2.fillPoly(depth_img, [pts], depth)
        else:
            cv2.polylines(rgb_img, [pts], True, color, 5)
            cv2.polylines(depth_img, [pts], True, depth, 5)

def apply_transformation_with_scaling(rgb_img, depth_img, rotation_angle=30, translation=(100, 50, 0.2), 
                                    reference_depth=1.0):
    """Apply rotation, translation, and depth-based scaling to the images"""
    height, width = rgb_img.shape[:2]
    
    # Calculate Z translation and scaling factor
    z_translation = translation[2]
    new_reference_depth = reference_depth + z_translation
    scale_factor = reference_depth / new_reference_depth  # Objects farther away appear smaller
    
    print(f"Scale factor based on depth change: {scale_factor:.3f}")
    print(f"Reference depth: {reference_depth} -> New depth: {new_reference_depth}")
    
    # Create transformation matrix with scaling
    center_x, center_y = width // 2, height // 2    # center of the image
    
    # Z translation involes scaling the image and adding a constant depth offset to the orignal depth values
    # First apply scaling
    scale_matrix = cv2.getRotationMatrix2D((center_x, center_y), 0, scale_factor)   # 2D affine transformation matrix for rotation and scaling: center of image / rotation angle / scale factor
    scaled_rgb = cv2.warpAffine(rgb_img, scale_matrix, (width, height))
    scaled_depth = cv2.warpAffine(depth_img, scale_matrix, (width, height))

    # Then adding a constant depth offset
    scaled_depth += z_translation

    # For debugging
    cv2.imshow("Scaled RGB image (Z translation)", scaled_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow("Scaled depth image (Z translation)", scaled_depth)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Then apply rotation and X-Y translation
    rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), rotation_angle, 1.0)
    rotation_matrix[0, 2] += translation[0] # X translation
    rotation_matrix[1, 2] += translation[1] # Y translation
    
    # Apply rotation and X-Y translation to scaled images
    transformed_rgb = cv2.warpAffine(scaled_rgb, rotation_matrix, (width, height))
    transformed_depth = cv2.warpAffine(scaled_depth, rotation_matrix, (width, height))

    # Add grid background to transformed image
    visualizer = Scene3DVisualizer()
    visualizer.draw_grid_background(transformed_rgb)
    
    # For debugging
    cv2.imshow("Transformed RGB image (rotation and X-Y translation)", transformed_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow("Transformed depth image (rotation and X-Y translation)", transformed_depth)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return transformed_rgb, transformed_depth, rotation_matrix, z_translation, scale_factor

def sift_matching(image1, image2, max_features=30):
    """Perform SIFT feature matching between two images"""
    # Convert to grayscale for SIFT
    img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    # Initialize SIFT detector with the requested number of features
    sift = cv2.SIFT_create(nfeatures=max_features)
    
    # Find keypoints and descriptors
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)
    
    if descriptors1 is None or descriptors2 is None:
        return [], [], []
    
    # Match descriptors using FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    
    # Apply Lowe's ratio test
    good_matches = []
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
    
    print(f"Number of good matches: {len(good_matches)}")
        
    # Create a blank image that can fit both images side by side
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Calculate the height of the combined image (maximum height of the two images)
    max_height = max(h1, h2)
    
    # Create a blank image with width = width of both images combined and height = max height
    combined_img = np.zeros((max_height, w1 + w2, 3), dtype=np.uint8)
    
    # Place the first image on the left side
    combined_img[0:h1, 0:w1] = image1
    
    # Place the second image on the right side
    combined_img[0:h2, w1:w1+w2] = image2
    
    # Draw circles only at keypoints involved in good matches
    for match in good_matches:
        # Get the coordinates of matching keypoints
        x1, y1 = map(int, keypoints1[match.queryIdx].pt)
        x2, y2 = map(int, keypoints2[match.trainIdx].pt)
        
        # Draw circle on the first image keypoint
        cv2.circle(combined_img, (x1, y1), 10, (0, 0, 255), 3)
        
        # Draw circle on the second image keypoint (shift x by w1)
        cv2.circle(combined_img, (x2 + w1, y2), 5, (0, 0, 255), 2)

    # Draw lines between matching keypoints
    for match in good_matches:  # only get keypoints that are good matches to draw lines
        # Get the coordinates of matching keypoints
        x1, y1 = map(int, keypoints1[match.queryIdx].pt)
        x2, y2 = map(int, keypoints2[match.trainIdx].pt)
        
        # Adjust x2 coordinate to account for the offset in the combined image
        x2 += w1
        
        # Draw a line between matching points (green color)
        cv2.line(combined_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
    
    # Display the combined image with keypoints and matches
    cv2.imshow('SIFT Matches', combined_img)
    
    # Wait for a key press to close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return keypoints1, keypoints2, good_matches

def extract_3d_points(keypoints, depth_img, camera_matrix):
    """Extract 3D points from 2D keypoints using depth information"""
    points_3d = []
    valid_indices = []
    
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    
    for i, kp in enumerate(keypoints):
        x, y = int(kp.pt[0]), int(kp.pt[1])
        
        if 0 <= x < depth_img.shape[1] and 0 <= y < depth_img.shape[0]:
            depth = depth_img[y, x]
            if depth > 0:
                # Convert to 3D coordinates
                z = depth
                x_3d = (x - cx) * z / fx
                y_3d = (y - cy) * z / fy
                points_3d.append([x_3d, y_3d, z])
                valid_indices.append(i)
    
    return np.array(points_3d), valid_indices

def estimate_pose_svd(points_3d_src, points_3d_dst):
    """Estimate pose using SVD (Kabsch algorithm)"""
    if len(points_3d_src) < 3 or len(points_3d_dst) < 3:
        return None, None
    
    # Center the point sets
    centroid_src = np.mean(points_3d_src, axis=0)
    centroid_dst = np.mean(points_3d_dst, axis=0)
    
    points_src_centered = points_3d_src - centroid_src
    points_dst_centered = points_3d_dst - centroid_dst
    
    # Compute cross-covariance matrix
    H = points_src_centered.T @ points_dst_centered
    
    # SVD
    U, S, Vt = np.linalg.svd(H)
    
    # Compute rotation matrix
    R_matrix = Vt.T @ U.T
    
    # Ensure proper rotation (determinant = 1)
    if np.linalg.det(R_matrix) < 0:
        Vt[-1, :] *= -1
        R_matrix = Vt.T @ U.T
    
    # Compute translation
    t_vector = centroid_dst - R_matrix @ centroid_src
    
    return R_matrix, t_vector

def create_camera_matrix(width, height, fov=60):
    """Create a simple camera matrix"""
    f = width / (2 * np.tan(np.radians(fov/2)))
    camera_matrix = np.array([
        [f, 0, width/2],
        [0, f, height/2],
        [0, 0, 1]
    ])
    return camera_matrix

def apply_pose_to_scene(rgb_img, depth_img, R_matrix, t_vector, camera_matrix):
    """Apply estimated pose transformation to create overlay"""
    height, width = rgb_img.shape[:2]
    
    # Create 3D point cloud from depth image
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    
    # Create meshgrid for all pixels
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    
    # Get valid depth points
    valid_mask = depth_img > 0
    u_valid = u[valid_mask]
    v_valid = v[valid_mask]
    depth_valid = depth_img[valid_mask]
    
    # Convert to 3D
    x_3d = (u_valid - cx) * depth_valid / fx
    y_3d = (v_valid - cy) * depth_valid / fy
    z_3d = depth_valid
    
    points_3d = np.column_stack([x_3d, y_3d, z_3d])
    
    # Apply transformation
    points_3d_transformed = (R_matrix @ points_3d.T).T + t_vector
    
    # Project back to 2D
    u_new = (points_3d_transformed[:, 0] * fx / points_3d_transformed[:, 2] + cx).astype(int)
    v_new = (points_3d_transformed[:, 1] * fy / points_3d_transformed[:, 2] + cy).astype(int)
    
    # Create transformed image
    transformed_img = np.zeros_like(rgb_img)
    
    # Map valid pixels
    valid_proj = (u_new >= 0) & (u_new < width) & (v_new >= 0) & (v_new < height)
    
    if np.any(valid_proj):
        transformed_img[v_new[valid_proj], u_new[valid_proj]] = rgb_img[v_valid[valid_proj], u_valid[valid_proj]]
    
    return transformed_img

def create_3d_visualization(original_rgb, original_depth, transformed_rgb, transformed_depth, 
                           kp1, kp2, matches, camera_matrix, R_estimated, t_estimated):
    """Create interactive 3D visualization with SIFT matches and estimated overlay"""
    from mpl_toolkits.mplot3d import Axes3D
    
    # Extract 3D points for matched keypoints
    matched_kp1 = [kp1[m.queryIdx] for m in matches]
    matched_kp2 = [kp2[m.trainIdx] for m in matches]
    
    points_3d_original, valid_orig = extract_3d_points(matched_kp1, original_depth, camera_matrix)
    points_3d_transformed, valid_trans = extract_3d_points(matched_kp2, transformed_depth, camera_matrix)
    
    # Ensure we have corresponding points
    min_points = min(len(points_3d_original), len(points_3d_transformed))
    if min_points > 0:
        points_3d_original = points_3d_original[:min_points]
        points_3d_transformed = points_3d_transformed[:min_points]
        
        # Create estimated points using SVD transformation
        if R_estimated is not None and t_estimated is not None:
            points_3d_estimated = (R_estimated @ points_3d_original.T).T + t_estimated
        else:
            points_3d_estimated = points_3d_original.copy()
        
        # Create 3D plot
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot original scene points (red)
        if len(points_3d_original) > 0:
            ax.scatter(points_3d_original[:, 0], points_3d_original[:, 1], points_3d_original[:, 2],
                      c='red', s=100, alpha=0.8, label=f'Original SIFT Points ({len(points_3d_original)})')
            
            # Add coordinate labels for original points
            for i, point in enumerate(points_3d_original[:10]):  # Show first 10 to avoid clutter
                ax.text(point[0], point[1], point[2], 
                       f'O{i}\n({point[0]:.2f},{point[1]:.2f},{point[2]:.2f})',
                       fontsize=8, color='red')
        
        # Plot transformed scene points (blue)
        if len(points_3d_transformed) > 0:
            ax.scatter(points_3d_transformed[:, 0], points_3d_transformed[:, 1], points_3d_transformed[:, 2],
                      c='blue', s=100, alpha=0.8, label=f'Transformed SIFT Points ({len(points_3d_transformed)})')
            
            # Add coordinate labels for transformed points
            for i, point in enumerate(points_3d_transformed[:10]):  # Show first 10 to avoid clutter
                ax.text(point[0], point[1], point[2], 
                       f'T{i}\n({point[0]:.2f},{point[1]:.2f},{point[2]:.2f})',
                       fontsize=8, color='blue')
        
        # Plot estimated points (light blue with transparency)
        if len(points_3d_estimated) > 0:
            ax.scatter(points_3d_estimated[:, 0], points_3d_estimated[:, 1], points_3d_estimated[:, 2],
                      c='lightblue', s=80, alpha=0.4, label=f'SVD Estimated Points ({len(points_3d_estimated)})')
            
            # Add coordinate labels for estimated points
            for i, point in enumerate(points_3d_estimated[:10]):  # Show first 10 to avoid clutter
                ax.text(point[0], point[1], point[2], 
                       f'E{i}\n({point[0]:.2f},{point[1]:.2f},{point[2]:.2f})',
                       fontsize=8, color='lightblue')
        
        # Draw lines connecting corresponding points
        if len(points_3d_original) > 0 and len(points_3d_transformed) > 0:
            for i in range(min(len(points_3d_original), len(points_3d_transformed), 20)):  # Limit to 20 lines
                # Original to transformed
                ax.plot([points_3d_original[i, 0], points_3d_transformed[i, 0]],
                       [points_3d_original[i, 1], points_3d_transformed[i, 1]],
                       [points_3d_original[i, 2], points_3d_transformed[i, 2]],
                       'g--', alpha=0.3, linewidth=1)
                
                # Original to estimated
                if len(points_3d_estimated) > i:
                    ax.plot([points_3d_original[i, 0], points_3d_estimated[i, 0]],
                           [points_3d_original[i, 1], points_3d_estimated[i, 1]],
                           [points_3d_original[i, 2], points_3d_estimated[i, 2]],
                           'orange', alpha=0.5, linewidth=2)
        
        # Set labels and title
        ax.set_xlabel('X Coordinate (m)')
        ax.set_ylabel('Y Coordinate (m)')
        ax.set_zlabel('Z Coordinate (m)')
        ax.set_title('3D SIFT Feature Correspondences and SVD Estimation\n(Interactive - Click and drag to rotate)')
        
        # Add legend
        ax.legend()
        
        # Set equal aspect ratio
        max_range = 0.5  # Adjust based on your scene scale
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([0, 1.5])
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        return fig, ax, points_3d_original, points_3d_transformed, points_3d_estimated
    else:
        print("No valid 3D points found for visualization")
        return None, None, None, None, None

def main():
    # Create scene
    visualizer = Scene3DVisualizer()
    original_rgb, original_depth = visualizer.create_scene()
    
    # Apply transformation with depth-based scaling
    reference_depth = 1.0  # Reference depth for scaling calculations
    transformed_rgb, transformed_depth, transform_matrix, z_translation, scale_factor = apply_transformation_with_scaling(
        original_rgb, original_depth, rotation_angle=25, translation=(80, 60, 0.5), reference_depth=reference_depth
    )
    
    print(f"\nApplied transformation: 25° rotation + (80, 60, 0.5) translation")
    
    # SIFT matching
    kp1, kp2, matches = sift_matching(original_rgb, transformed_rgb)
    
    print(f"\nFound {len(matches)} good matches")
    
    # Create camera matrix
    camera_matrix = create_camera_matrix(visualizer.width, visualizer.height)
    
    # Extract 3D points for matched keypoints
    matched_kp1 = [kp1[m.queryIdx] for m in matches]
    matched_kp2 = [kp2[m.trainIdx] for m in matches]
    
    points_3d_src, valid_src = extract_3d_points(matched_kp1, original_depth, camera_matrix)
    points_3d_dst, valid_dst = extract_3d_points(matched_kp2, transformed_depth, camera_matrix)
    
    # Filter matches to only include those with valid 3D points
    valid_matches = []
    valid_3d_src = []
    valid_3d_dst = []
    
    for i, (src_idx, dst_idx) in enumerate(zip(valid_src, valid_dst)):
        if i < len(valid_dst) and src_idx < len(points_3d_src):
            valid_matches.append(matches[src_idx])
            valid_3d_src.append(points_3d_src[i])
            if i < len(points_3d_dst):
                valid_3d_dst.append(points_3d_dst[i])
    
    valid_3d_src = np.array(valid_3d_src)
    valid_3d_dst = np.array(valid_3d_dst[:len(valid_3d_src)])  # Ensure same length
    
    print(f"Valid 3D matches: {len(valid_3d_src)}")
    
    if len(valid_3d_src) >= 3:
        # Estimate pose using SVD
        print("Estimating pose using SVD...")
        R_estimated, t_estimated = estimate_pose_svd(valid_3d_src, valid_3d_dst)
        
        if R_estimated is not None:
            print("Pose estimation successful!")
            print(f"Rotation matrix:\n{R_estimated}")
            print(f"Translation vector: {t_estimated}")
            
            # Create 3D visualization
            fig_3d, ax_3d, points_orig, points_trans, points_est = create_3d_visualization(
                original_rgb, original_depth, transformed_rgb, transformed_depth,
                kp1, kp2, matches, camera_matrix, R_estimated, t_estimated
            )
            
            # Apply estimated transformation to create overlay
            overlay_img = apply_pose_to_scene(original_rgb, original_depth, 
                                            R_estimated, t_estimated, camera_matrix)
            
            # Create comprehensive visualization
            fig_main, axes = plt.subplots(3, 3, figsize=(20, 16))
            
            # Row 1: Original scene and depth
            axes[0, 0].imshow(cv2.cvtColor(original_rgb, cv2.COLOR_BGR2RGB))
            axes[0, 0].set_title('Original RGB Scene\n(with grid background)', fontsize=12)
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(original_depth, cmap='gray')
            axes[0, 1].set_title('Original Depth Map', fontsize=12)
            axes[0, 1].axis('off')
            
            # Show transformed scene with scaling info
            axes[0, 2].imshow(cv2.cvtColor(transformed_rgb, cv2.COLOR_BGR2RGB))
            axes[0, 2].set_title(f'Transformed Scene\n(Scale: {scale_factor:.3f}, Z+{z_translation:.2f})', fontsize=12)
            axes[0, 2].axis('off')
            
            # Row 2: SIFT matches and depth analysis
            if len(matches) > 0:
                match_img = cv2.drawMatches(gray_original, kp1, gray_transformed, kp2, 
                                          matches[:20], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                axes[1, 0].imshow(match_img, cmap='gray')
                axes[1, 0].set_title(f'SIFT Matches (showing 20/{len(matches)})', fontsize=12)
            else:
                axes[1, 0].text(0.5, 0.5, 'No matches found', ha='center', va='center', 
                               transform=axes[1, 0].transAxes, fontsize=14)
                axes[1, 0].set_title('No SIFT Matches', fontsize=12)
            axes[1, 0].axis('off')
            
            # Show transformed depth
            transformed_depth_vis = ((transformed_depth - transformed_depth.min()) / 
                                   (transformed_depth.max() - transformed_depth.min() + 1e-6) * 255).astype(np.uint8)
            axes[1, 1].imshow(transformed_depth_vis, cmap='gray')
            axes[1, 1].set_title('Transformed Depth Map', fontsize=12)
            axes[1, 1].axis('off')
            
            # Show depth difference
            depth_diff = np.abs(transformed_depth - original_depth)
            depth_diff_vis = ((depth_diff - depth_diff.min()) / 
                            (depth_diff.max() - depth_diff.min() + 1e-6) * 255).astype(np.uint8)
            axes[1, 2].imshow(depth_diff_vis, cmap='hot')
            axes[1, 2].set_title('Depth Difference\n(Hot = More Change)', fontsize=12)
            axes[1, 2].axis('off')
            
            # Row 3: SVD results and overlay comparison
            axes[2, 0].imshow(cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB))
            axes[2, 0].set_title('SVD Pose Applied\nto Original Scene', fontsize=12)
            axes[2, 0].axis('off')
            
            # Side-by-side comparison
            axes[2, 1].imshow(cv2.cvtColor(transformed_rgb, cv2.COLOR_BGR2RGB))
            axes[2, 1].set_title('Actual Transformed Scene', fontsize=12)
            axes[2, 1].axis('off')
            
            # Create overlay comparison
            alpha = 0.6
            overlay_comparison = cv2.addWeighted(transformed_rgb, alpha, overlay_img, 1-alpha, 0)
            axes[2, 2].imshow(cv2.cvtColor(overlay_comparison, cv2.COLOR_BGR2RGB))
            axes[2, 2].set_title('Overlay Comparison\n(SVD vs Actual)', fontsize=12)
            axes[2, 2].axis('off')
            
            plt.tight_layout()
            plt.show()
            
            # Print detailed analysis
            print(f"\n=== Detailed Analysis ===")
            print(f"Original scene objects at various depths: 0.2 to 0.8")
            print(f"Applied Z-translation: +{z_translation:.3f}m")
            print(f"Depth-based scale factor: {scale_factor:.3f}")
            print(f"This means objects appear {(1/scale_factor):.2f}x {'larger' if scale_factor < 1 else 'smaller'} due to depth change")
            print(f"SIFT detected {len(kp1)} features in original, {len(kp2)} in transformed")
            print(f"Good matches: {len(matches)}")
            print(f"3D correspondences used for SVD: {len(valid_3d_src)}")
            
            if points_orig is not None and len(points_orig) > 0:
                print(f"\n=== Sample 3D Coordinates ===")
                print(f"Original SIFT Points (first 3):")
                for i, point in enumerate(points_orig[:3]):
                    print(f"  Point O{i}: X={point[0]:.3f}, Y={point[1]:.3f}, Z={point[2]:.3f}")
                
                print(f"\nTransformed SIFT Points (first 3):")
                for i, point in enumerate(points_trans[:3]):
                    print(f"  Point T{i}: X={point[0]:.3f}, Y={point[1]:.3f}, Z={point[2]:.3f}")
                
                print(f"\nSVD Estimated Points (first 3):")
                for i, point in enumerate(points_est[:3]):
                    print(f"  Point E{i}: X={point[0]:.3f}, Y={point[1]:.3f}, Z={point[2]:.3f}")
        else:
            print("Pose estimation failed - not enough valid points")
    else:
        print("Not enough valid 3D matches for pose estimation (need at least 3)")

if __name__ == "__main__":
    main()
