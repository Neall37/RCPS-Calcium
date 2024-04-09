import json
import matplotlib.pyplot as plt
from numpy import array, zeros
from tifffile import imread, imwrite
from glob import glob
import numpy as np
from PCAICA import process_calcium_imaging_data
import cv2
from tqdm import tqdm


def apply_fourier_transform(frame):
    #Convert the frame to float type for more precision
    frame_float = frame.astype(float)

    #Calculate the mean and standard deviation of the frame
    mean = np.mean(frame_float)
    std = np.std(frame_float)

    #Standardize the frame
    standardized_frame = ((frame_float - mean) / std).astype(int)

    # Apply Fourier Transform
    f = np.fft.fft2(standardized_frame)
    fshift = np.fft.fftshift(f)

    return fshift

def high_pass_filter(size, cutoff):
    rows, cols = size
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.uint8)
    mask[crow-cutoff:crow+cutoff, ccol-cutoff:ccol+cutoff] = 0
    return mask


def fourier_pass(img):
    fshift = apply_fourier_transform(img)

    # Apply high-pass filter
    high_pass = high_pass_filter(fshift.shape, cutoff=1)  # Cutoff frequency can be adjusted

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

def output(imgs):
    # Initialize the maximum pixel value holder
    max_pixel_values = None

    for frame in tqdm(imgs):
        # Process the frame with Fourier transform
        fshift = apply_fourier_transform(frame)

        # Apply high-pass filter (if desired, adjust or remove this step)
        high_pass = high_pass_filter(fshift.shape, cutoff=3)
        fshift_filtered = fshift * high_pass

        # Applying the inverse Fourier Transform
        f_ishift = np.fft.ifftshift(fshift_filtered)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)

        # Initialize or update the max_pixel_values array
        if max_pixel_values is None:
            max_pixel_values = img_back
        else:
            max_pixel_values = np.maximum(max_pixel_values, img_back)

    # After processing all frames, max_pixel_values contains the max pixel across all frames
    plt.imshow(max_pixel_values, cmap='gray')
    plt.title('Max Pixel Value Across Frames')
    plt.show()

    return max_pixel_values

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

#
def tomask(coords,dims):
    mask = zeros(dims)
    # Unpack the zipped coordinates into separate lists for each dimension
    y_coords, x_coords = zip(*coords)
    # Convert lists of coordinates into numpy arrays for indexing
    mask[np.array(y_coords), np.array(x_coords)] = 1
    return mask

if __name__ == '__main__':
    # # load the images
    files = sorted(glob('images/*.tiff'))
    imgs = array([imread(f) for f in files])
    # imwrite('imgs.tif', imgs)
    dims = imgs.shape[1:]


    ## load the regions (training data only)
    with open('regions/regions.json') as f:
        regions = json.load(f)

    masks = array([tomask(s['coordinates'],dims) for s in regions])
    # print(len(masks))

    # index = 3  # Change this to display a different image/mask pair
    # print("imgs shape:", imgs.shape)
    # # Clip the data
    frame_indices = np.random.choice(imgs.shape[0], 3048, replace=False)
    movie = imgs[frame_indices, :, :]

    # Get summary for movie
    # img_back = output(imgs)
    # imwrite('After_ft.tif',img_back)

    ## Get Maximum Value summary and FT
    # max_across_time = np.max(movie, axis=0)
    # plt.imshow(max_across_time, cmap='gray')  # 'cmap' sets the color map to grayscale
    # plt.title('Maximum Value Across Time')
    # plt.colorbar()  # Optionally add a colorbar to indicate the scale
    # plt.show()
    # img_back = fourier_pass(max_across_time)
    # imwrite('After_fourier.tif', img_back)

    # Cell Detection Using PCA-ICA
    print(np.mean(movie, axis=-1, keepdims=True))
    binary_matrix, neuron_positions, neuron_masks = cell_detection(movie)
    # # Define the file path where you want to save the data
    filename = 'positions.json'
    # Convert keys from int64 to str for JSON serialization
    cell_positions_str_keys = {str(key): value for key, value in neuron_positions.items()}
    # Serialize and save the dictionary
    with open(filename, 'w') as file:
        json.dump(cell_positions_str_keys, file)
    # show the outputs
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(binary_matrix, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(masks.sum(axis=0), cmap='gray')
    plt.show()


    # # Signle Frame
    # img = imread('images/image00002.tiff')
    #
    # plt.imshow(img, cmap='gray')
    # plt.show()
    #
    # img_back = fourier_pass(img)
    # imwrite('After_fourier.tif', img_back)
