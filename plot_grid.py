import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from tifffile import imread
import json

HIT_COLOR = np.array([255, 255, 255])
MISSED_COLOR = np.array([255, 69, 85])
MISFIRE_COLOR = np.array([64, 181, 188])


def tomask(masks, dims):
    video_mask = []
    # Iterate over each mask in the dataset
    for i in range(len(masks)):
        image_mask = np.zeros(dims)
        regions = masks[i]["regions"]
        for region in regions:
            coords = region["coordinates"]
            image_mask[tuple(zip(*coords))] = 1
        video_mask.append(image_mask)
    return video_mask


def plot_grid(list_img_list, list_result_list, output_dir):
    nrows = len(list_result_list)
    ncols = len(list_img_list[0])
    print(nrows)
    print(ncols)

    if nrows <= ncols:
        fig, axs = plt.subplots(nrows=2, ncols=ncols, figsize=(ncols*10, 20))
        for i in range(nrows):
            for j in range(ncols):
                ax1 = axs[0] if nrows == 1 else axs[0, j]
                ax2 = axs[1] if nrows == 1 else axs[1, j]
                ax1.axis('off')
                ax1.imshow(list_img_list[i][j], aspect='equal', cmap='gray')
                ax2.axis('off')
                ax2.imshow(list_result_list[i][j], aspect='equal')
    else:
        fig, axs = plt.subplots(nrows=2*nrows, ncols=ncols, figsize=(ncols*10, 10*2*nrows))
        for i in range(nrows):
            for j in range(ncols):
                ax1 = axs[2*i] if ncols == 1 else axs[2*i, j]
                ax2 = axs[2*i+1] if ncols == 1 else axs[2*i+1, j]
                ax1.axis('off')
                ax1.imshow(list_img_list[i][j], aspect='equal')
                ax2.axis('off')
                ax2.imshow(list_result_list[i][j], aspect='equal')

    plt.tight_layout()
    plt.show()
    plt.savefig(output_dir + 'multicell_grid_fig.svg')


def get_results(lhats, nc_list, val_imgs, val_masks):
    list_img_list = list()
    list_result_list = list()
    num_plot = len(lhats)

    for j in range(len(nc_list)):
        nc = nc_list[j]
        filter_bool = [True] * len(val_imgs) 

        val_imgs_nc = val_imgs[filter_bool]
        val_masks_nc = val_masks[filter_bool]

        img_list = list()
        result_list = list()

        for i, lhat in enumerate(lhats):
            Tlamhat = val_imgs_nc[0] >= -lhat

            val_masks_nc = (np.array(val_masks_nc) > 0).astype(float)
            result = val_masks_nc[0]
            result[result == 0] = -2
            result = result - Tlamhat.astype(float)
            print(result.shape)

            result_display = np.zeros((result.shape[0], result.shape[1], 3))
            result_display[result == 0] = HIT_COLOR / 255.
            result_display[result == -3] = MISFIRE_COLOR / 255.
            result_display[result == 1] = MISSED_COLOR / 255.

            result_list.append(result_display)

            img = val_imgs_nc[0]
            img_list.append(resize(img, (result_display.shape[0], result_display.shape[1])))

        list_img_list.append(img_list)
        list_result_list.append(result_list)

    print(len(list_img_list))
    return list_img_list, list_result_list, num_plot


if __name__ == '__main__':
    imgs = imread('test.tif')
    dims = imgs.shape[1:]

    with open('label_masks_001.json') as f:
        masks = json.load(f)

    video_masks = tomask(masks, dims)
    output_dir = 'outputs/'
    frame_indices = 2
    start = 0
    video_masks = np.array(video_masks)
    movie = imgs[start:start + frame_indices, :, :]
    video_masks = video_masks[start:start + frame_indices, :, :]

    lhats = [-0.5015525563489085, -0.5078620448724628, -0.512895735250646, -0.5166670232906727,
             -0.519418237990555, -0.5231178557825974, -0.52685670774923, -0.5302474595031418,
             -0.5341270491493565, -0.5374325352813044, -0.5411623001540307, -0.5454122962744384,
             -0.5485784204762778, -0.5514057657398778, -0.55444336161674, -0.5582381409936266,
             -0.5624508968020118, -0.5670625533161511, -0.5725216138873643, -0.5795862027673103,
             -0.5890695886389795, -0.5992223843956419, -0.6076109401668066, -0.6166665678434053,
             -0.6270424232650996, -0.6409777794148948, -0.6540758560170282, -0.6684872027562812,
             -0.684399034313736, -0.7032845986272013]
    stepsize = 3
    lhats = lhats[::stepsize]
    nc_list = [1, 2]

    list_img_list, list_result_list, num_plot = get_results(lhats, nc_list, movie, video_masks)

    plot_grid(list_img_list, list_result_list, output_dir)
