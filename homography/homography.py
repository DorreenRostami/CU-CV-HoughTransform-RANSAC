import cv2
import numpy as np
from skimage.feature import match_descriptors, ORB, plot_matches
import matplotlib.pyplot as plt

def matchPics(I1, I2):
    # Given two images I1 and I2, perform SIFT matching to find candidate match pairs

    ### YOUR CODE HERE
    ### You can use skimage or OpenCV to perform SIFT matching 
    image8bit1 = cv2.normalize(I1, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    image8bit2 = cv2.normalize(I2, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(image8bit1, None)
    kp2, des2 = sift.detectAndCompute(image8bit2, None)
    
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    matches_bf = bf.match(des1,des2)
    matches_bf = sorted(matches_bf, key = lambda x:x.distance)
    matches_bf = matches_bf[:170] #keep 170 of the best matches (matches with least distances)

    # img3 = cv2.drawMatches(image8bit1, kp1, image8bit2, kp2, matches_bf, image8bit2, flags=2)
    # plt.imshow(img3),plt.show()

    locs1 = np.zeros((len(matches_bf), 2), dtype=int)
    locs2 = np.zeros((len(matches_bf), 2), dtype=int)
    matches = np.zeros((len(matches_bf), 2), dtype=int)
    for i, match in enumerate(matches_bf):
        locs1[i] = kp1[match.queryIdx].pt
        locs2[i] = kp2[match.trainIdx].pt
        matches[i, 0] = matches[i, 1] = i

    locs1 = np.flip(locs1, 1) #flip x and y
    locs2 = np.flip(locs2, 1)
    ### END YOUR CODE
    
    return matches, locs1, locs2

def computeH_ransac(matches, locs1, locs2):

    # Compute the best fitting homography using RANSAC given a list of matching pairs
    
    ### YOUR CODE HERE
    ### You should implement this function using Numpy only
    itr_count = 100000 
    inlier_eps = 40
    best_count = 0

    #Convert to homogeneous coordinates
    locs1_homo = np.column_stack((locs1, np.ones((len(locs1), 1))))

    for _ in range(itr_count):
        #Select four feature pairs (at random)
        rand_idx = np.random.choice(len(matches), 4, replace=False)
        img1_pts = locs1[matches[rand_idx, 0]]
        img2_pts = locs2[matches[rand_idx, 1]]

        #Compute homography H
        homography = compute_homography(img1_pts, img2_pts)

        transformed_pts = np.dot(homography, locs1_homo.T).T
        divided_pts = transformed_pts[:,:2] / transformed_pts[:, 2][:, np.newaxis] #[wx, wy, w] -> [wx/w, wy/w]

        #Compute inliers where SSD(pi’, Hpi)< ε
        distances = np.sqrt(np.sum((locs2 - divided_pts) ** 2, axis=1))
        inlier_distances = distances < inlier_eps
        inlier_count = np.sum(inlier_distances)

        #Keep largest set of inliers
        if inlier_count > best_count:
            best_count = inlier_count
            inliers = np.where(inlier_distances)[0] #index of where it was true

    #Re-compute least-squares H estimate on all of the inliers
    bestH = compute_homography(locs1[inliers], locs2[inliers])    
    # print(best_count)

    ### END YOUR CODE
    
    return bestH, inliers

def compute_homography(img1_pts, img2_pts):
    A = []
    for pt1, pt2 in zip(img1_pts, img2_pts):
        y, x = pt1
        yp, xp = pt2
        A.append([-x, -y, -1, 0, 0, 0, x * xp, y * xp, xp]) #week5 slide58
        A.append([0, 0, 0, -x, -y, -1, x * yp, y * yp, yp])
    A = np.array(A)
    U, S, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    return H

def compositeH(H, template, img):
    # Create a compositie image after warping the template image on top
    # of the image using homography
    template = cv2.normalize(template, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

    #Create mask of same size as template
    mask = np.ones((template.shape[0], template.shape[1]), dtype=np.uint8)*255

    #Warp mask by appropriate homography
    warped_mask = cv2.warpPerspective(mask, H, (img.shape[1], img.shape[0]))

    #Warp template by appropriate homography
    warped_template = cv2.warpPerspective(template, H, (img.shape[1], img.shape[0]))

    #Use mask to combine the warped template and the image
    warped_mask_3ch = cv2.merge([warped_mask, warped_mask, warped_mask])
    composite_img = np.where(warped_mask_3ch > 0, warped_template, img) #where mask is non-zero (is 255) use harry potter, otherwise use original image
    
    return composite_img
