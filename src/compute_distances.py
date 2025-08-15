import numpy as np
from scipy.linalg import sqrtm

# Calculate the Bhattacharyya distance between the two distributions
def bhattacharyya_distance(normals_1, normals_2):
    """
    Calculates the Bhattacharyya distance between the normal distributions of two surfaces.

    Args:
        normals_1 (np.ndarray): An array of normal vectors for surface 1.
        normals_2 (np.ndarray): An array of normal vectors for surface 2.

    Returns:
        float: The Bhattacharyya distance between the two distributions.
    """
    mean1 = np.mean(normals_1, axis=0)
    cov1 = np.cov(normals_1, rowvar=False)

    mean2 = np.mean(normals_2, axis=0)
    cov2 = np.cov(normals_2, rowvar=False)

    cov_avg = 0.5 * (cov1 + cov2)
    mean_diff = mean1 - mean2
    
    term1 = 0.125 * mean_diff.T @ np.linalg.inv(cov_avg) @ mean_diff
    
    det_cov1 = np.linalg.det(cov1)
    det_cov2 = np.linalg.det(cov2)
    det_cov_avg = np.linalg.det(cov_avg)

    term2 = 0.5 * np.log(det_cov_avg / np.sqrt(det_cov1 * det_cov2))

    bhattacharyya_dist = term1 + term2

    return bhattacharyya_dist

def jensen_shannon_divergence(normals_1, normals_2):
    """
    Calculates the Jensen-Shannon Divergence between two normal distributions.

    Args:
        normals_1 (np.ndarray): An array of normal vectors for surface 1.
        normals_2 (np.ndarray): An array of normal vectors for surface 2.

    Returns:
        float: The Jensen-Shannon Divergence.
    """
    # Helper function for KL Divergence
    def kl_divergence_mvn(mean_p, cov_p, mean_q, cov_q):
        """Calculates the KL Divergence from distribution P to Q for MVNs."""
        k = len(mean_p)
        cov_q_inv = np.linalg.inv(cov_q)
        
        trace_term = np.trace(cov_q_inv @ cov_p)
        mean_diff = mean_q - mean_p
        mahalanobis_term = mean_diff.T @ cov_q_inv @ mean_diff
        log_term = np.log(np.linalg.det(cov_q) / np.linalg.det(cov_p))
        
        kl_div = 0.5 * (trace_term + mahalanobis_term - k + log_term)
        return kl_div

    # Estimate distribution parameters
    mean1 = np.mean(normals_1, axis=0)
    cov1 = np.cov(normals_1, rowvar=False)
    
    mean2 = np.mean(normals_2, axis=0)
    cov2 = np.cov(normals_2, rowvar=False)

    # Calculate the average distribution (mean is simple average, but covariance is not)
    # This is a common approximation for this type of problem.
    mean_avg = 0.5 * (mean1 + mean2)
    cov_avg = 0.5 * (cov1 + cov2)
    
    # Calculate KL divergence from each distribution to the average distribution
    kl1 = kl_divergence_mvn(mean1, cov1, mean_avg, cov_avg)
    kl2 = kl_divergence_mvn(mean2, cov2, mean_avg, cov_avg)
    
    # Jensen-Shannon Divergence is the average of the two KL divergences
    js_divergence = 0.5 * (kl1 + kl2)
    
    return js_divergence

def hellinger_distance(normals_1, normals_2):
    """
    Calculates the Hellinger distance between the normal distributions of two surfaces.

    Args:
        normals_1 (np.ndarray): An array of normal vectors for surface 1.
        normals_2 (np.ndarray): An array of normal vectors for surface 2.

    Returns:
        float: The Hellinger distance.
    """
    mean1 = np.mean(normals_1, axis=0)
    cov1 = np.cov(normals_1, rowvar=False)

    mean2 = np.mean(normals_2, axis=0)
    cov2 = np.cov(normals_2, rowvar=False)

    cov_avg = 0.5 * (cov1 + cov2)
    mean_diff = mean1 - mean2
    
    term1 = 0.125 * mean_diff.T @ np.linalg.inv(cov_avg) @ mean_diff
    
    det_cov1 = np.linalg.det(cov1)
    det_cov2 = np.linalg.det(cov2)
    det_cov_avg = np.linalg.det(cov_avg)

    term2 = 0.5 * np.log(det_cov_avg / np.sqrt(det_cov1 * det_cov2))

    bhattacharyya_dist = term1 + term2
    
    # The Hellinger distance is derived from the Bhattacharyya coefficient
    # For Gaussian distributions, this is a known relationship
    hellinger_dist = np.sqrt(1 - np.exp(-bhattacharyya_dist))

    return hellinger_dist

def mahalanobis_distance(normals_1, normals_2):
    """
    Calculates the Mahalanobis distance between the means of two normal distributions.
    
    Args:
        normals_1 (np.ndarray): An array of normal vectors for surface 1.
        normals_2 (np.ndarray): An array of normal vectors for surface 2.

    Returns:
        float: The Mahalanobis distance.
    """
    mean1 = np.mean(normals_1, axis=0)
    cov1 = np.cov(normals_1, rowvar=False)

    mean2 = np.mean(normals_2, axis=0)
    cov2 = np.cov(normals_2, rowvar=False)
    
    # Use the average covariance matrix for the distance
    cov_avg = 0.5 * (cov1 + cov2)
    cov_avg_inv = np.linalg.inv(cov_avg)
    
    mean_diff = mean1 - mean2
    
    # Mahalanobis distance squared
    mahalanobis_sq = mean_diff.T @ cov_avg_inv @ mean_diff
    
    return np.sqrt(mahalanobis_sq)

def wasserstein_distance_O2(normals_1, normals_2):
    """
    Calculates the 2-Wasserstein distance between two normal distributions.
    
    Args:
        normals_1 (np.ndarray): An array of normal vectors for surface 1.
        normals_2 (np.ndarray): An array of normal vectors for surface 2.

    Returns:
        float: The 2-Wasserstein distance.
    """
    mean1 = np.mean(normals_1, axis=0)
    cov1 = np.cov(normals_1, rowvar=False)

    mean2 = np.mean(normals_2, axis=0)
    cov2 = np.cov(normals_2, rowvar=False)
    
    mean_diff = mean1 - mean2
    
    # Calculate the mean component
    mean_term = np.sum(mean_diff**2)
    
    # Calculate the covariance component using a numerical method for sqrt of matrix
    cov_sqrt = sqrtm(cov1 @ cov2)
    cov_term = np.trace(cov1 + cov2 - 2 * cov_sqrt)
    
    return np.sqrt(mean_term + cov_term)

