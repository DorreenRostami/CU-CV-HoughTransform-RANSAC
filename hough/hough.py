# import other necessary libaries
from utils import create_line, create_mask
from skimage import io, feature
import matplotlib.pylab as plt
import numpy as np


def main():
    # load the input image
    img = io.imread('road.jpg', as_gray=True)

    # run Canny edge detector to find edge points
    edges = feature.canny(img) #sigma=1 (default value) has an output similar to the one in the handout

    # create a mask for ROI by calling create_mask
    mask = create_mask(img.shape[0], img.shape[1])

    # extract edge points in ROI by multipling edge map with the mask
    masked_img = mask * edges

    # perform Hough transform
    accumulator, thetas, rhos = hough_line(masked_img)

    # find the right lane by finding the peak in hough space
    max_accum_idx = np.argmax(accumulator) #this returns a flattened index 
    max_accum_idx_unraveled = np.unravel_index(max_accum_idx, accumulator.shape) #unravel index to it's corresponding row and col
    max_rho = rhos[max_accum_idx_unraveled[0]]
    max_theta = thetas[max_accum_idx_unraveled[1]]
    blue_line_xs, blue_line_ys = create_line(max_rho, max_theta, img)

    # zero out the values in accumulator around the neighborhood of the peak
    neighborhood_radius = 530  #radius of the neighborhood found with trial and error
    accumulator[max_accum_idx_unraveled] = 0  #set peak value to zero
    for i in range(len(accumulator)):
        for j in range(len(accumulator[i])):
            if np.abs(i - max_accum_idx_unraveled[0]) < neighborhood_radius:
                accumulator[i, j] = 0

    # find the left lane by finding the peak in hough space
    max_accum_idx = np.argmax(accumulator) #this returns a flattened index 
    max_accum_idx_unraveled = np.unravel_index(max_accum_idx, accumulator.shape) #unravel index to it's corresponding row and col
    max_rho = rhos[max_accum_idx_unraveled[0]]
    max_theta = thetas[max_accum_idx_unraveled[1]]
    orange_line_xs, orange_line_ys = create_line(max_rho, max_theta, img)

    # plot the results
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 4))
    ax[0][0].imshow(edges, cmap='gray')
    ax[0][0].set_title(r'Canny with, $\sigma=1$', fontsize=20)
    ax[0][0].axis("off")
    ax[0][1].imshow(mask, cmap='gray')
    ax[0][1].set_title("Mask", fontsize=20)
    ax[0][1].axis("off")
    ax[1][0].imshow(masked_img, cmap='gray')
    ax[1][0].set_title("Edges in ROI", fontsize=20)
    ax[1][0].axis("off")
    ax[1][1].imshow(img, cmap='gray')
    ax[1][1].plot(blue_line_xs, blue_line_ys, color='blue')
    ax[1][1].plot(orange_line_xs, orange_line_ys, color='orange')
    ax[1][1].set_title("Result", fontsize=20)
    ax[1][1].axis("off")
    plt.show()


def hough_line(img):
    #rho and theta ranges
    h, w = img.shape
    max_rho = int(np.ceil(np.sqrt(h*h + w*w)))  #maximum possible rho value is the diagonal length of the image
    rho_range = np.linspace(-max_rho, max_rho, max_rho * 2)
    theta_range = np.deg2rad(np.arange(-90, 90))  #theta from -90 to 90 degrees
    # hough_space = np.zeros((2 * max_rho, len(theta_range)), dtype=np.uint64)

    accumulator = np.zeros((2 * max_rho, len(theta_range)), dtype=np.uint64) #hough accumulator of theta vs rho
    y_idxs, x_idxs = np.nonzero(img)  #edge pixels

    #vote in the accumulator
    for i in range(len(y_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        for theta_i, theta in enumerate(theta_range):
            rho = int(np.round(x * np.cos(theta) + y * np.sin(theta))) + max_rho
            accumulator[rho, theta_i] += 1

    return accumulator, theta_range, rho_range

if __name__ == "__main__":
    main()