import open3d as o3d
import numpy as np

# --- Data Loading and Point Cloud Estimation ---

def get_point_cloud_from_off(file_path, num_points):
    """
    Loads a 3D mesh from an OFF file, samples a point cloud, and returns it
    as a NumPy array.

    Args:
        file_path (str): Path to the OFF file.
        num_points (int): The number of points to sample from the mesh.

    Returns:
        np.ndarray: A NumPy array of shape (num_points, 3) representing the
                    point cloud, or None if the file could not be loaded.
    """
    mesh = o3d.io.read_triangle_mesh(file_path)

    if mesh.is_empty():
        print(f"Error: Could not load the mesh from {file_path}")
        return None
    
    # Use Poisson disk sampling for a high-quality point cloud
    pcd = mesh.sample_points_poisson_disk(number_of_points=num_points)

    # Convert the Open3D PointCloud object to a NumPy array
    point_cloud = np.asarray(pcd.points)

    return point_cloud, pcd

# --- Normal Estimation and Statistical Embedding ---

def estimate_normals_from_pc(pcd_array):
    """
    Estimates surface normals from a point cloud array.

    Args:
        pcd_array (np.ndarray): A NumPy array of shape (N, 3) representing the
                                point cloud.

    Returns:
        np.ndarray: A NumPy array of shape (N, 3) representing the normal
                    vectors, or None if the point cloud is empty.
    """
    if pcd_array.size == 0:
        return None

    # Convert NumPy array to Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_array)

    # Estimate normals for each point
    pcd.estimate_normals()

    # Get the normals as a NumPy array
    normals = np.asarray(pcd.normals)

    return normals

def calculate_mean_covariance(normals):
    """
    Calculates the mean vector and covariance matrix from a set of normal vectors.

    Args:
        normals (np.ndarray): A NumPy array of shape (N, 3) with normal vectors.

    Returns:
        tuple: A tuple containing the mean vector (np.ndarray) and the
               covariance matrix (np.ndarray).
    """
    # Calculate the mean vector (mu)
    mu = np.mean(normals, axis=0)
    
    # Calculate the covariance matrix (Sigma)
    sigma = np.cov(normals, rowvar=False)

    return mu, sigma

# --- Main Object Processing ---

def process_object_embedding(file_path, num_points):
    """
    Processes a single OFF file to generate its statistical embedding.

    Args:
        file_path (str): Path to the OFF file.
        num_points (int): The number of points to sample from the mesh.

    Returns:
        tuple: A tuple with the mean vector and covariance matrix, or None
               if processing failed.
    """
    # Step 1: Get point cloud
    pc = get_point_cloud_from_off(file_path, num_points)
    if pc is None:
        return None

    # Step 2: Estimate surface normals
    normals = estimate_normals_from_pc(pc)
    if normals is None:
        return None
        
    # Step 3: Calculate the statistical embedding (mean and covariance)
    mu, sigma = calculate_mean_covariance(normals)

    return mu, sigma

