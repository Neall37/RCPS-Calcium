import json
import matplotlib.pyplot as plt
from numpy import array, zeros
from tifffile import imread, imwrite
from glob import glob
import numpy as np
from PCA-ICA import process_calcium_imaging_data
import cv2
from tqdm import tqdm
from scipy.special import softmax
from skimage.morphology import label as connected_components
from skimage.morphology import reconstruction
from skimage.filters import gaussian
from skimage.measure import regionprops


def find_peaks(sigmoid):
    sigmoid = gaussian(sigmoid,0.5)
    seed = np.copy(sigmoid)
    seed[1:-1, 1:-1] = sigmoid.min()
    mask = sigmoid
    dilated = reconstruction(seed, mask, method='dilation')
    peaks = (sigmoid - dilated)
    binarized_peaks = peaks > 0.05
    labels, num_components = connected_components(binarized_peaks, background=0, return_num=True, connectivity=2)
    proposals = regionprops(labels, intensity_image=None, cache=True)
    normalization_value = np.ones_like(peaks)
    minsize = 25
    for region in proposals:
        # take regions with large enough areas
        if region.area >= minsize:
        # draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox
            minr = max(minr-minsize, 0)
            minc = max(minc-minsize, 0)
            maxr = min(maxr+minsize, normalization_value.shape[0]-1)
            maxc = min(maxc+minsize, normalization_value.shape[1]-1)
            np.minimum(normalization_value[minr:maxr, minc:maxc], peaks[minr:maxr, minc:maxc].max(), out = normalization_value[minr:maxr, minc:maxc])
    peaks = np.maximum(sigmoid, peaks/normalization_value)

    return peaks


def apply_fourier_transform(frame):
    #Convert the frame to float type for more precision
    frame_float = frame.astype(float)

    #Calculate the mean and standard deviation of the frame
    mean = np.mean(frame_float)
    std = np.std(frame_float)

    standardized_frame = ((frame_float - mean) / std).astype(int)

    # Apply Fourier Transform
    f = np.fft.fft2(standardized_frame)
    fshift = np.fft.fftshift(f)

    return fshift


def high_pass_filter(shape, cutoff):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2

    mask = np.zeros((rows, cols), np.float32)

    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
            if distance > cutoff:
                mask[i, j] = 1

    return mask


def fourier_pass(img):
    fshift = apply_fourier_transform(img)

    # Apply high-pass filter
    high_pass = high_pass_filter(fshift.shape, cutoff=3)  # Cutoff frequency can be adjusted

    fshift_filtered = fshift * high_pass
    # Applying the inverse Fourier Transform
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    # # Display the resulting image
    plt.imshow(img_back, cmap='gray')
    plt.title('Inverse FFT of Averaged Frames')
    plt.show()
    return img_back

def softmax_sci(x, axis=0):
    # x_normalized = (x - np.mean(x)) / np.std(x)
    return softmax(x, axis=axis)

def output(imgs):
    regions = np.zeros((1000, imgs.shape[1], imgs.shape[2]))
    k = 0
    # Initialize the maximum pixel value holder
    max_pixel_values = None

    for frame in tqdm(imgs):
        frame = find_peaks(frame)
        fshift = apply_fourier_transform(frame)

        high_pass = high_pass_filter(fshift.shape, cutoff=3)
        fshift_filtered = fshift * high_pass

        # Applying the inverse Fourier Transform
        f_ishift = np.fft.ifftshift(fshift_filtered)
        img_back = np.fft.ifft2(f_ishift)
        T = 3
        img_back = np.abs(img_back)/T
        img_back = 1 / (1 + np.exp(-img_back))
        regions[k, :, :] = img_back
        k += 1

    return regions

def cell_detection(movie):
    n_components = 10
    activity_threshold = 0.9  # Top 10% of activity
    mu = 0.5  # Equal weight for spatial and temporal information
    min_area = 20
    neuron_positions, neuron_masks = process_calcium_imaging_data(movie, min_area, n_components, activity_threshold, mu)

    # Visualize the neuron positions
    height, width = movie.shape[1], movie.shape[2]
    binary_matrix = np.zeros((height, width))
    for cell in neuron_positions:
        for pos in cell:
            y, x = pos
            binary_matrix[y, x] = 1

    return binary_matrix, neuron_positions, neuron_masks



def tomask(coords, dims):
    mask = zeros(dims)
    y_coords, x_coords = zip(*coords)
    # Convert lists of coordinates into numpy arrays for indexing
    mask[np.array(y_coords), np.array(x_coords)] = 1
    return mask

def get_mask(img, regions, dims, adjacent_size=4):
    mean = np.mean(img)
    mask = np.zeros(dims, dtype=np.float32)  # Initialize the mask with zeros
    mask_regions = []

    for i, region in enumerate(regions):
        coords = region['coordinates']
        y_coords, x_coords = zip(*coords)
        y_coords = np.array(y_coords)
        x_coords = np.array(x_coords)

        y_min = max(np.min(y_coords) - adjacent_size, 0)
        y_max = min(np.max(y_coords) + adjacent_size + 1, dims[0])
        x_min = max(np.min(x_coords) - adjacent_size, 0)
        x_max = min(np.max(x_coords) + adjacent_size + 1, dims[1])

        region_pixels = img[y_coords, x_coords]
        adjacent_pixels = img[y_min:y_max, x_min:x_max]

        # Create a mask for the adjacent pixels
        y_grid, x_grid = np.meshgrid(np.arange(y_min, y_max), np.arange(x_min, x_max), indexing='ij')
        region_mask = np.zeros_like(adjacent_pixels, dtype=bool)
        region_mask[y_coords - y_min, x_coords - x_min] = True

        adjacent_pixels = adjacent_pixels[~region_mask]

        # Calculate the mean values of the region and its adjacent area
        region_mean = np.mean(region_pixels)
        adjacent_mean = np.mean(adjacent_pixels)
      
        if region_mean >= 1.7 * adjacent_mean and region_mean >= mean:
            mask[y_coords, x_coords] = 1
            mask_regions.append({
                'region': i,
                'coordinates': list(zip(y_coords.tolist(), x_coords.tolist()))
            })

    plt.imshow(mask, cmap='gray')
    plt.show()

    return mask, mask_regions



if __name__ == '__main__':
    files = sorted(glob('images/*.tiff'))
    imgs = array([imread(f) for f in files])

    with open('regions/regions.json') as f:
        regions = json.load(f)
 
    img_back = output(movie)

    plt.imshow(img_back[2], cmap='gray')  # 'cmap' sets the color map to grayscale
    plt.title('Maximum Value Across Time')
    plt.colorbar()  # Optionally add a colorbar to indicate the scale
    plt.show()
