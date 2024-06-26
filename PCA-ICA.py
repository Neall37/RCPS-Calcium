import numpy as np
from sklearn.decomposition import IncrementalPCA, FastICA
from scipy.ndimage import gaussian_filter
from skimage.measure import label, regionprops
from tifffile import imread, imwrite
import matplotlib.pyplot as plt
from tqdm import tqdm

def register_and_normalize(movie):
    # Calculate the mean across the time axis
    movie_mean = np.mean(movie, axis=-1, keepdims=True)

    # Use a small constant to prevent division by zero
    epsilon = 1e-10

    # Normalize the signal in each pixel
    normalized_movie = (movie - movie_mean) / (movie_mean + epsilon)

    # Subtract the mean fluorescence from each time frame
    normalized_movie -= np.mean(normalized_movie, axis=(0, 1), keepdims=True)

    return normalized_movie


def efficient_pca(movie, n_components, batch_size=100):
    print(f"Initial movie shape: {movie.shape}")
    n_pixels, n_frames = movie.shape

    # Initialize IncrementalPCA
    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)

    # Fit IncrementalPCA on the data
    print("Fitting IncrementalPCA...")
    for i in tqdm(range(0, n_pixels, batch_size), desc='Processing batches'):
        batch = movie[i:i + batch_size, :]
        ipca.partial_fit(batch)

    V = ipca.components_.T  # Transpose to match the expected shape (n_frames, n_components)

    print("Computing spatial filters U...")
    U = np.dot(movie, V)

    explained_variance = ipca.explained_variance_
    S = np.sqrt(explained_variance * n_frames)  # Singular values approximation

    return U, S, V

def marchenko_pastur_pdf(x, Q, sigma=1):
    lambda_plus = sigma * (1 + np.sqrt(1 / Q))**2
    lambda_minus = sigma * (1 - np.sqrt(1 / Q))**2
    return Q / (2 * np.pi * sigma * x) * np.sqrt((lambda_plus - x) * (x - lambda_minus))


def gaussian_noise_reference_variances(n_frames, n_pixels, n_components=200):
    Q = n_frames / float(n_pixels)
    sigma = 1  # Assuming unit variance of the noise
    lambda_plus = sigma * (1 + np.sqrt(1 / Q)) ** 2
    lambda_minus = sigma * (1 - np.sqrt(1 / Q)) ** 2
    lambdas = np.linspace(lambda_minus, lambda_plus, n_components)
    mp_variances = marchenko_pastur_pdf(lambdas, Q) * (lambda_plus - lambda_minus) / n_components
    return lambdas, mp_variances

def visualize_spatial_filters(U, height, width, n_components_to_show=10):
    fig, axs = plt.subplots(1, n_components_to_show, figsize=(20, 2))
    for i in range(n_components_to_show):
        axs[i].imshow(U[:, i].reshape(height, width), cmap='gray')
        axs[i].axis('off')
        axs[i].set_title(f'PC {i+1}')
    plt.show()

def select_pcs(U, S, V, n_frames, n_pixels, height, width, auto_select=True, Klow=0, visualize_filters=True):
    variances = S ** 2
    print(n_pixels)
    print(height)
    _, noise_variances = gaussian_noise_reference_variances(n_frames, n_pixels, len(variances))
    print("U shape:", U.shape)
    print("S shape:", S.shape)
    print("V shape:", V.shape)
    if auto_select:
        # Automatic Khigh selection based on where real variances deviate from noise variances
        diff = variances - noise_variances
        Khigh = np.argmax(diff <= 0) if np.any(diff <= 0) else len(variances)
    else:
        Khigh = len(variances) 

    print("Khigh: ", Khigh)

    # Visualization for manual Klow selection
    if visualize_filters:
        if height is None or width is None:
            raise ValueError("height and width must be provided for visualization.")
        # Visualize the spatial filters
        print("Visualizing spatial filters for manual Klow selection...")
        visualize_spatial_filters(U, height, width)
        # Here you could manually adjust Klow based on the visual inspection
        try:
            Klow = int(input("Enter the new Klow after visual inspection: "))
        except ValueError:
            print("Invalid Klow input; using original Klow value.")

    # Adjusted truncation based on potentially new Klow value
    U_truncated = U[:, Klow:Khigh]
    S_truncated = S[Klow:Khigh]
    V_truncated = V[Klow:Khigh, :]

    Mwhite = U_truncated @ np.diag(S_truncated) @ V_truncated
    return Mwhite, U_truncated, S_truncated, V_truncated

def dimensional_reduction(movie, n_components):

    # Reshape the movie to a 2D matrix (pixels x time)
    width = movie.shape[2]
    height = movie.shape[1]
    reshaped_movie = movie.reshape(movie.shape[0], -1).T

    n_pixels = reshaped_movie.shape[0]
    n_frames = reshaped_movie.shape[1]

    U, S, V = efficient_pca(reshaped_movie, n_components)
    Mwhite, U_truncated, S_truncated, V_truncated = select_pcs(U, S, V, n_frames, n_pixels, height,width)
    return Mwhite, U_truncated, S_truncated, V_truncated


def spatio_temporal_ica(U, S, V, mu=0.5):
    n_pixels, n_components = U.shape
    n_frames = V.shape[1]
    print("U shape_truncated:", U.shape)
    print("S shape_truncated:", S.shape)
    print("V shape_truncated:", V.shape)
    # Normalize U and V based on S (singular values)
    U_norm = U * S[np.newaxis, :]
    V_norm = V * S[:, np.newaxis]
    print("U_norm shape:", U_norm.shape)
    print("V_norm.T shape:", V_norm.T.shape)
    # Concatenate spatial and temporal components
    Y = np.vstack((mu * U_norm, (1 - mu) * V_norm.T))
    print("Y shape:", Y.shape)
    print("Y.T shape:", Y.T.shape)
    # Perform ICA
    ica = FastICA(n_components=n_components, max_iter=3000, tol=1e-3)
    W = ica.fit_transform(Y.T).T  # Transpose W to match dimensions with Y

    # Separate spatial filters and temporal signals from the transformed components
    spatial_filters = W[:n_components, :]
    temporal_signals = W[n_components:, :]

    print("Spatial filters shape:", spatial_filters.shape)
    return spatial_filters, temporal_signals

def reconstruct_spatial_maps(U, spatial_filters, height, width):
    reconstructed_maps = []
    for i in range(spatial_filters.shape[0]):
        reconstructed_map = np.dot(U, spatial_filters[i, :])
        reconstructed_map = reconstructed_map.reshape((height, width))
        # Normalize reconstructed map to [0, 1] range for display
        reconstructed_map = (reconstructed_map - np.min(reconstructed_map)) / (np.max(reconstructed_map) - np.min(reconstructed_map))
        reconstructed_maps.append(reconstructed_map)

    return reconstructed_maps


def segment_spatial_filters(spatial_filter, sigma=1.5, threshold_std=1.5, min_area=100):
    smoothed_filter = gaussian_filter(spatial_filter, sigma=sigma)

    # Threshold to create a binary mask
    mean_intensity = np.mean(smoothed_filter)
    std_intensity = np.std(smoothed_filter)
    threshold = mean_intensity + threshold_std * std_intensity
    binary_mask = smoothed_filter > threshold

    labeled_image = label(binary_mask)  # Only capture the labeled image

    segmented_filters = []
    for region in regionprops(labeled_image):
        if region.area < min_area:
            continue

        segmented_filter = np.zeros_like(spatial_filter, dtype=bool)
        for coord in region.coords:
            segmented_filter[coord[0], coord[1]] = True  # Use True to mark the neuron's mask

        segmented_filters.append(segmented_filter)

    return segmented_filters


def process_calcium_imaging_data(movie, min_area, n_components=200, activity_threshold=0.9, mu=0.5):
    normalized_movie = register_and_normalize(movie)
    Mwhite, U_truncated, S_truncated, V_truncated = dimensional_reduction(normalized_movie, n_components)
    spatial_filters, temporal_signals = spatio_temporal_ica(U_truncated, S_truncated, V_truncated, mu)
    height, width = movie.shape[1], movie.shape[2]
    spatial_maps = reconstruct_spatial_maps(U_truncated, spatial_filters, height, width)

    neuron_positions_lists = []  # List of lists, each containing the positions within a neuron mask
    neuron_masks = []  # List of binary masks for each neuron

    for spatial_map in spatial_maps:
        segmented_filters = segment_spatial_filters(spatial_map, sigma=1.5, threshold_std=1.5, min_area=min_area)
        for segmented_filter in segmented_filters:
            # Use the labeled image to find connected components
            labeled_image, _ = label(segmented_filter, return_num=True)
            regions = regionprops(labeled_image)

            for region in regions:
                # Extract all coordinates for the current neuron mask
                coords = region.coords  # Array of shape (num_pixels, 2) for each position

                # Convert coordinates to a list of tuples (y, x) and add to the list
                neuron_positions_lists.append([(coord[0], coord[1]) for coord in coords])

                neuron_mask = np.zeros_like(segmented_filter, dtype=bool)
                neuron_mask[coords[:, 0], coords[:, 1]] = True
                neuron_masks.append(neuron_mask)

    return neuron_positions_lists, neuron_masks
if __name__ == '__main__':
    # Example usage
    movie = imread("M0_2017-10-13_002.tif")  # Load the calcium imaging movie data

    n_components = 20
    activity_threshold = 0.9  # Top 10% of activity
    mu = 0.5  # Equal weight for spatial and temporal information
    min_area = 50
    small_movie = movie
    neuron_positions, neuron_masks = process_calcium_imaging_data(small_movie, min_area, n_components, activity_threshold, mu)

    # Visualize the neuron positions
    height, width = small_movie.shape[1], small_movie.shape[2]
    binary_matrix = np.zeros((height, width))
    for cell in neuron_positions:
        for pos in cell:
            y, x = pos
            binary_matrix[y, x] = 1

    plt.figure(figsize=(8, 8))
    plt.imshow(binary_matrix, cmap='gray', interpolation='nearest')
    plt.title(f"Binary Plot of Neuron Positions")
    plt.axis('off')
    plt.show()
