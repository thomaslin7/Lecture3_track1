import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
            {'position': (180, 120), 'depth': 0.2, 'color': [170, 200, 50], 'text': 'AFA'},
            {'position': (450, 400), 'depth': 0.4, 'color': [50, 170, 220], 'text': 'Project'},
            {'position': (210, 380), 'depth': 0.6, 'color': [0, 255, 255], 'text': 'Course'},
            {'position': (450, 160), 'depth': 0.8, 'color': [20, 150, 255], 'text': 'Innovation'},
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
        """Draw a word (text) on the image with color and apply depth."""
        text = obj['text']
        position = obj['position']
        color = obj['color']
        depth = obj['depth']

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2
        thickness = 3

        # Draw text on RGB image
        cv2.putText(rgb_img, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
        
        # Bounding box for depth image
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        x, y = position
        cv2.rectangle(depth_img, (x, y - text_height), (x + text_width, y), depth, -1)
    
def apply_transformation(rgb_img, depth_img, rotation_angle=30, translation=(100, 50, 0.3)):
    """Apply rotation, translation with depth to the image"""
    height, width = rgb_img.shape[:2]
    
    # Get translation in z
    z_translation = translation[2]
    # Get center of the image
    center_x, center_y = width // 2, height // 2

    # Z translation
    depth_img += z_translation
    
    # Apply rotation and X-Y translation
    rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), rotation_angle, 1.0)
    rotation_matrix[0, 2] += translation[0] # X translation
    rotation_matrix[1, 2] += translation[1] # Y translation
    
    # Apply rotation and X-Y translation to scaled images
    transformed_rgb = cv2.warpAffine(rgb_img, rotation_matrix, (width, height))
    transformed_depth = cv2.warpAffine(depth_img, rotation_matrix, (width, height))

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

    return transformed_rgb, transformed_depth

def sift_matching(image1, image2, max_features=100):
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
        cv2.circle(combined_img, (x1, y1), 5, (0, 0, 255), 2)
        
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
        cv2.line(combined_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
    
    # Display the combined image with keypoints and matches
    cv2.imshow('SIFT Matches', combined_img)
    
    # Wait for a key press to close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return keypoints1, keypoints2, good_matches

def extract_3d_points(keypoints, depth_img):
    """Extract 3D points from 2D keypoints using depth information"""
    points_3d = np.zeros((len(keypoints), 3)) # Initialize 3D points array with zeros
    points_3d[:, :2] = keypoints    # Copy 2D keypoints to the first two columns of points_3d
    
    for i, pt in enumerate(keypoints):
        x, y = int(pt[0]), int(pt[1])
        points_3d[i, 2] = depth_img[y, x]
    
    return points_3d

def estimate_3d_transform(src_3d, dst_3d):
    """Estimate 3D transformation using SVD (Kabsch algorithm)"""
    if len(src_3d) < 3 or len(dst_3d) < 3:
        return None, None
    
    # Center the point sets
    src_centroid = np.mean(src_3d, axis=0)
    dst_centroid = np.mean(dst_3d, axis=0)
    
    src_centered = src_3d - src_centroid
    dst_centered = dst_3d - dst_centroid
    
    # Compute cross-covariance matrix
    H = src_centered.T @ dst_centered
    
    # SVD (Singular Value Decomposition)
    U, S, Vt = np.linalg.svd(H)
    
    # Compute rotation matrix
    R_matrix = Vt.T @ U.T
    
    # Ensure proper rotation (determinant = 1)
    if np.linalg.det(R_matrix) < 0:
    # Checks if the computed rotation matrix R is a reflection instead of a proper rotation
    # A proper rotation matrix should have det(R) = +1, not -1
        Vt[-1, :] *= -1
        # Flips the sign of the last row of Vt, this corrects the reflection problem
        R_matrix = Vt.T @ U.T
        # Recomputes the corrected rotation matrix
    
    # Compute translation vector
    t_vector = dst_centroid - R_matrix @ src_centroid
    
    return R_matrix, t_vector

def show_estimated_transformation(original_rgb, transformed_rgb, R, t, alpha=0.5):
    """
    Apply the estimated transformation to the original image and overlay it on the transformed image
    """
    
    # Convert 3x3 rotation matrix to 2x2 (take top-left 2x2 part)
    R_2d = R[:2, :2]
    
    # Convert 3D translation to 2D (take first 2 components)
    t_2d = t[:2]
    
    # Create 2x3 transformation matrix for cv2.warpAffine
    # Format: [[R11, R12, tx], [R21, R22, ty]]
    transform_matrix = np.hstack([R_2d, t_2d.reshape(-1, 1)])
    
    # Apply transformation to original image
    height, width = original_rgb.shape[:2]
    estimated_rgb = cv2.warpAffine(original_rgb, transform_matrix, (width, height))
    
    # Create overlay
    overlay_result = cv2.addWeighted(transformed_rgb, 1.0, estimated_rgb, alpha, 0)
    
    # Display overlay
    cv2.imshow('Transformation Overlay', overlay_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def visualize_3d_points(src_3d, dst_3d, estimated_dst_3d):
    """Visualize the 3D points in a 3D coordinate system, with scaled depth"""
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Scale Z-coordinate by 300 (for visualization purposes only)
    src_3d[:, 2] *= 300
    dst_3d[:, 2] *= 300
    estimated_dst_3d[:, 2] *= 300

    # Plot source points in red
    ax.scatter(src_3d[:, 0], src_3d[:, 1], src_3d[:, 2], 
              c='red', marker='o', s=80, alpha=0.8, label='Source Points (src_3d)')
    
    # Plot destination points in green
    ax.scatter(dst_3d[:, 0], dst_3d[:, 1], dst_3d[:, 2], 
              c='green', marker='o', s=80, alpha=0.8, label='Destination Points (dst_3d)')
    
    # Plot estimated destination points in blue
    ax.scatter(estimated_dst_3d[:, 0], estimated_dst_3d[:, 1], estimated_dst_3d[:, 2], 
              c='blue', marker='o', s=80, alpha=0.8, label='Estimated Destination Points (estimated_dst_3d)')
    
    # # Draw lines connecting corresponding points for better visualization
    # for i in range(len(src_3d)):
    #     # Line from source to actual destination (red to green)
    #     ax.plot([src_3d[i, 0], dst_3d[i, 0]], 
    #             [src_3d[i, 1], dst_3d[i, 1]], 
    #             [src_3d[i, 2], dst_3d[i, 2]], 
    #             'k--', alpha=0.3, linewidth=1)
        
    #     # Line from actual destination to estimated destination (green to blue)
    #     ax.plot([dst_3d[i, 0], estimated_dst_3d[i, 0]], 
    #             [dst_3d[i, 1], estimated_dst_3d[i, 1]], 
    #             [dst_3d[i, 2], estimated_dst_3d[i, 2]], 
    #             'orange', alpha=0.6, linewidth=2)
    
    # Set labels and title
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    ax.set_zlabel('Z Coordinate (Depth)', fontsize=12)
    ax.set_title('3D Point Transformation Visualization', fontsize=14, fontweight='bold')
    
    # Add legend
    ax.legend(loc='upper left', fontsize=10)
    
    # Add grid for better visualization
    ax.grid(True, alpha=0.3)
    
    # Set equal aspect ratio for better visualization
    max_range = np.array([src_3d.max()-src_3d.min(), 
                         dst_3d.max()-dst_3d.min(), 
                         estimated_dst_3d.max()-estimated_dst_3d.min()]).max() / 2.0
    
    mid_x = (src_3d[:, 0].max() + src_3d[:, 0].min()) / 2.0
    mid_y = (src_3d[:, 1].max() + src_3d[:, 1].min()) / 2.0
    mid_z = (src_3d[:, 2].max() + src_3d[:, 2].min()) / 2.0
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Show the plot
    plt.tight_layout()
    plt.show()

def main():
    # Create scene
    visualizer = Scene3DVisualizer()
    original_rgb, original_depth = visualizer.create_scene()
    
    # Apply transformation with depth
    transformed_rgb, transformed_depth = apply_transformation(
        original_rgb, original_depth, rotation_angle=25, translation=(80, 60, 0.5)
    )
    
    print(f"\nApplied transformation: 25° rotation + (80, 60, 0.5) translation")
    
    # SIFT matching wtih FLANN
    kp1, kp2, matches = sift_matching(original_rgb, transformed_rgb)
    
    # Extract the corresponding keypoints of good matches
    matched_kp1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    matched_kp2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
    
    # Extract z coordinates from the depth images
    src_3d = extract_3d_points(matched_kp1, original_depth)
    dst_3d = extract_3d_points(matched_kp2, transformed_depth)

    # Get the translation and rotation matrices for estimated 3D transformation
    R, t = estimate_3d_transform(src_3d, dst_3d)

    # Apply estimated transformation to the original RGB image
    show_estimated_transformation(original_rgb, transformed_rgb, R, t, alpha=0.3)

    # Implement the SVD-based estimated transformation
    estimated_dst_3d = (R @ src_3d.T).T + t

    # Visualize the 3D points in a coordinate system, with scaled depth
    visualize_3d_points(src_3d, dst_3d, estimated_dst_3d)

if __name__ == "__main__":
    main()