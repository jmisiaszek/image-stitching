import cv2
from matplotlib import pyplot as plt
import numpy as np
import glob
import os

# Marker map for the 6x4 grid of markers and their 3D coordinates.
marker_map = {
    29: [
        [0., 406., 0.],
        [0., 238., 0.],
        [168., 238., 0.],
        [168., 406., 0.],
    ],
    28: [
        [0., 168., 0.],
        [0., 0., 0.],
        [168., 0., 0.],
        [168., 168., 0.],
    ],
    24: [
        [238., 406., 0.],
        [238., 238., 0.],
        [406., 238., 0.],
        [406., 406., 0.],
    ],
    23: [
        [238., 168., 0.],
        [238., 0., 0.],
        [406., 0., 0.],
        [406., 168., 0.],
    ],
    19: [
        [476., 406., 0.],
        [476., 238., 0.],
        [644., 238., 0.],
        [644., 406., 0.],
    ],
    18: [
        [476., 168., 0.],
        [476., 0., 0.],
        [644., 0., 0.],
        [644., 168., 0.],
    ],
}

# Calculate the reprojection error for a set of object and image points.
def reprojection_error(obj_points, img_points, rvecs, tvecs, camera_matrix, dist_coeffs):
    total_error = 0
    total_points = 0
    for i in range(len(obj_points)):
        img_points2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = np.linalg.norm(img_points[i] - img_points2.reshape(-1, 2))
        total_error += np.sum(error)
        total_points += len(img_points[i])
    return total_error / total_points

# Calibrate the camera using all available information.
def calibrate_six():
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    images = glob.glob("calibration/*.png")

    obj_points = []
    img_points = []

    for image in images:
        img = cv2.imread(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        corners, ids, rejected = detector.detectMarkers(gray)
        if corners is not None:
            obj_points_for_image = []
            img_points_for_image = []
            for id, corner in zip(ids, corners):
                obj_points_for_image.append(np.array(marker_map[id[0]], dtype=np.float32))
                img_points_for_image.append(corner[0]) 

            obj_points.append(np.vstack(obj_points_for_image))
            img_points.append(np.vstack(img_points_for_image)) 

            # cv2.aruco.drawDetectedMarkers(img, corners, ids)
            # cv2.imshow("Detected Tags", img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
    obj_points = np.array(obj_points)
    obj_points = obj_points.reshape(-1, 24, 3)

    img_points = np.array(img_points)
    img_points = img_points.reshape(-1, 24, 2)

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, gray.shape[::-1], None, None
    )

    error = reprojection_error(obj_points, img_points, rvecs, tvecs, camera_matrix, dist_coeffs)

    h, w = gray.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), alpha=0)

    for image in images:
        img = cv2.imread(image)
        undistorted = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)

        # cv2.imshow("Original Image", img)
        # cv2.imshow("Undistorted Image", undistorted)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    return error

# Calibrate the camera using only one tag and reuse the same picture six times.
def calibrate_one():
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    images = glob.glob("calibration/img*.png")

    obj_points = []
    img_points = []

    for image in images:
        img = cv2.imread(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        corners, ids, rejected = detector.detectMarkers(gray)
        if corners is not None:
            obj_points_for_image = []
            img_points_for_image = []
            for id, corner in zip(ids, corners):
                obj_points_for_image.append(np.array(marker_map[28], dtype=np.float32))
                img_points_for_image.append(corner[0]) 

        obj_points.append(np.vstack(obj_points_for_image))
        img_points.append(np.vstack(img_points_for_image)) 

    obj_points = np.array(obj_points)
    obj_points = obj_points.reshape(-1, 4, 3)

    img_points = np.array(img_points)
    img_points = img_points.reshape(-1, 4, 2)

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, gray.shape[::-1], None, None
    )

    error = reprojection_error(obj_points, img_points, rvecs, tvecs, camera_matrix, dist_coeffs)

    h, w = gray.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), alpha=0)

    os.makedirs("./undistorted", exist_ok=True)

    for image in glob.glob("stitching/img*.png"):
        img = cv2.imread(image)
        undistorted = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)

        filename = os.path.basename(image)
        save_path = f"./undistorted/{filename}"
        cv2.imwrite(save_path, undistorted)

        # cv2.imshow("Original Image", img)
        # cv2.imshow("Undistorted Image", undistorted)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    return error

# Calibrate the camera using both methods and compare the reprojection errors.
def calibrate_and_compare():
    error1 = calibrate_six()
    print(f"Reprojection error using all available markers: {error1}")
    error2 = calibrate_one()
    print(f"Reprojection error reusing same picture six times: {error2}")


# Project an image using a homography matrix.
def apply_proj(img, H, output_h, output_w):
    h, w = img.shape[:2]
    H_inv = np.linalg.inv(H)
    trans_img = np.zeros((output_h, output_w, 3), dtype=np.uint8)

    for y in range(output_h):
        for x in range(output_w):
            dest = np.array([x, y, 1], dtype=np.float32)
            src = H_inv @ dest
            src = src / (src[2] ) # Avoid division by zero.

            src_x, src_y = int(round(src[0])), int(round(src[1]))
            if 0 <= src_x < w and 0 <= src_y < h:
                trans_img[y, x] = img[src_y, src_x]
    
    return trans_img

# Simple test for image projection.
def test_proj():
    img = cv2.imread("undistorted/img1.png")
    H = np.array([
        [0.8, 0.3, 50],
        [-0.3, 1.2, 30],
        [0.0001, 0.0002, 1]
    ])
    trans_img = apply_proj(img, H, img.shape[0], img.shape[1])
    print(img.shape)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Transformed Image")
    plt.imshow(cv2.cvtColor(trans_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.show()


# Find the transformation matrix between two sets of points.
def find_transform(points_src, points_dst):
    assert len(points_src) == len(points_dst)
    num_points = len(points_src)
    A = np.zeros((2 * num_points, 9))

    for i in range(num_points):
        x, y = points_src[i]
        u, v = points_dst[i]

        A[2 * i] = np.array([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A[2 * i + 1] = np.array([0, 0, 0, -x, -y, -1, v * x, v * y, v])

    _, _, V = np.linalg.svd(A)
    H = V[-1].reshape(3, 3)
    H = H / H[2, 2]
    return H

# Test the transformation matrix calculation.
def test_transform():
    np.random.seed(42)

    for i in range(10):
        H = np.random.rand(3, 3)
        H[-1, -1] = 1

        points_src = np.random.rand(4, 2) * 100
        points_src_3d = np.hstack([points_src, np.ones((4, 1))])
        points_dst = (H @ points_src_3d.T).T
        points_dst = points_dst[:, :2] / points_dst[:, 2, None]

        H_est = find_transform(points_src, points_dst)
        # H_est = H_est / H_est[-1, -1]

        assert np.allclose(H, H_est, atol=1e-3)
        print(f"Test {i + 1} passed.")


# Transform an image using hand-picked points.
def manual_transform():
    img1 = cv2.imread('./undistorted/img1.png')
    img2 = cv2.imread('./undistorted/img2.png')

    points_src = np.array([
        [294, 287], # Orange rectangle
        [515, 375], # White rectangle
        [417, 181], # Blue circle
        [932, 361]  # Shelf corner
    ])

    points_dst = np.array([
        [417, 298],
        [625, 382],
        [531, 195],
        [1049, 368]
    ])

    H = find_transform(points_src, points_dst)

    h, w = img1.shape[:2]
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]])
    new_corners = []

    min_x = np.inf
    min_y = np.inf
    max_x = -np.inf
    max_y = -np.inf

    for [x, y] in corners:
        new_corner = H @np.array([x, y, 1], dtype=np.float32) 
        new_corner /= new_corner[2]
        new_x, new_y = int(round(new_corner[0])), int(round(new_corner[1]))
        min_x = min(min_x, new_x)
        min_y = min(min_y, new_y)
        max_x = max(max_x, new_x)
        max_y = max(max_y, new_y)
        new_corners.append([new_x, new_y])

    T = np.array([
        [1, 0, -min_x],
        [0, 1, -min_y],
        [0, 0, 1]
    ])

    H = T @ H

    img1_transformed = apply_proj(img1, H, max_y - min_y, max_x - min_x)

    # Showing the results
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    
    plt.axis('off')
    plt.title("Image 2")
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.subplot(1, 2, 2)

    plt.axis('off')
    plt.title("Image 1 Transformed")
    plt.imshow(cv2.cvtColor(img1_transformed, cv2.COLOR_BGR2RGB))
    plt.show()


# Applies transformation H to img2 and projects both images on the common canvas.
def project_canvas(img1, img2, H):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    corners = np.array([[0, 0], [w1, 0], [w1, h1], [0, h1]])
    new_corners = []
    for [x, y] in corners:
        new_corner = H @ np.array([x, y, 1], dtype=np.float32)
        new_corner /= new_corner[2]
        new_corners.append([int(round(new_corner[0])), int(round(new_corner[1]))])
    
    corners2 = np.array([[0, 0], [w2, 0], [w2, h2], [0, h2]])
    corners = np.concatenate([new_corners, corners2])

    min_x = np.min(corners[:, 0])
    min_y = np.min(corners[:, 1])
    max_x = np.max(corners[:, 0])
    max_y = np.max(corners[:, 1])

    T = np.array([
        [1, 0, -min_x],
        [0, 1, -min_y],
        [0, 0, 1]
    ], dtype=np.float32)

    H_img1 = T @ H
    H_img2 = T

    canvas = np.zeros((max_y - min_y, max_x - min_x, 3), dtype=np.uint8)

    img1_transformed = apply_proj(img1, H_img1, canvas.shape[0], canvas.shape[1])
    img2_transformed = apply_proj(img2, H_img2, canvas.shape[0], canvas.shape[1])

    return img1_transformed, img2_transformed

# Applies color correction as in the blog post.
def color_correction(images, gamma=2.2):
    alphas = [np.ones(3)]
    num_images = len(images)

    for i in range(1, num_images):
        prev_img = images[i - 1]
        cur_img = images[i]

        overlap = np.logical_and(prev_img.any(axis=-1), cur_img.any(axis=-1))
        if not overlap.any():
            alphas.append(np.ones(3))
            continue

        overlap_prev = prev_img[overlap]
        overlap_cur = cur_img[overlap]

        sum_prev = np.sum(overlap_prev**gamma, axis=0)
        sum_cur = np.sum(overlap_cur**gamma, axis=0)

        sum_cur = np.where(sum_cur == 0, 1e-6, sum_cur)

        alphas.append(sum_prev/sum_cur)

    alphas = np.array(alphas)
    gc = np.sum(alphas, axis=0)/np.sum(alphas**2, axis=0)

    corrected_images = []
    for i in range(num_images):
        alpha = alphas[i]
        coeff = (gc * alpha) ** (1 / gamma)

        corrected_img = np.clip(images[i] * coeff, 0, 255).astype(np.uint8)
        corrected_images.append(corrected_img)
    
    return corrected_images

# Calculate error for finding the seam.
def error_matrix(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    return (gray1 - gray2) ** 2

# Find optimal seam with dp.
def optimal_seam(img1, img2):
    overlap = np.logical_and(img1.any(axis=-1), img2.any(axis=-1))
    error = error_matrix(img1, img2)

    overlap_rows = np.where(overlap.any(axis=1))[0]
    min_row = overlap_rows[0]
    max_row = overlap_rows[-1]
    h_overlap = max_row - min_row + 1
    w = error.shape[1]

    error = error[min_row:max_row + 1, :]
    overlap = overlap[min_row:max_row + 1, :]

    # h, w = img1.shape[:2]
    dp = np.full_like(error, np.inf, dtype=np.float32)
    path = np.full_like(dp, -1, dtype=np.int32)

    dp[0, :] = np.where(overlap[0, :], error[0, :], np.inf)

    for i in range(1, h_overlap):
        for j in range(w):
            if not overlap[i, j]:
                continue
                
            prev_dp = np.inf
            best_w = j
            for prev_w in [j - 1, j, j + 1]:
                if prev_w < 0 or prev_w >= w or not overlap[i - 1, prev_w]:
                    continue
                if dp[i - 1, prev_w] < prev_dp:
                    prev_dp = dp[i - 1, prev_w]
                    best_w = prev_w
            
            if prev_dp < np.inf:
                dp[i, j] = prev_dp + error[i, j]
                path[i, j] = best_w
    
    valid_last_row = dp[-1][overlap[-1]]

    min_val = np.min(valid_last_row)
    j = np.where(dp[-1] == min_val)[0][0]
    i = h_overlap - 1

    seam = []
    while i >= 0:
        row = i + min_row
        seam.append((row, j))
        j = path[i, j]
        i -= 1

    return seam[::-1]

# Blend images along the seam.
def blend_images(img1, img2, seam):
    blended = img2.copy()
    for i, j in seam:
        blended[i, j:] = img1[i, j:]
    return blended

# Crop out the black border.
def crop(img):
    mask = img.any(axis=-1)
    coords = np.argwhere(mask)

    min_y, min_x = coords.min(axis=0)
    max_y, max_x = coords.max(axis=0) + 1

    return img[min_y:max_y, min_x:max_x]

# Task 5 - stitch images with hand picked points.
def run_project_canvas():
    img1 = cv2.imread('./undistorted/img1.png')
    img2 = cv2.imread('./undistorted/img2.png')

    points_src = np.array([
        [294, 287], # Orange rectangle
        [515, 375], # White rectangle
        [417, 181], # Blue circle
        [932, 361]  # Shelf corner
    ])

    points_dst = np.array([
        [417, 298],
        [625, 382],
        [531, 195],
        [1049, 368]
    ])

    H = find_transform(points_src, points_dst)
    imgs = project_canvas(img1, img2, H)

    imgs = color_correction(imgs)
    img1, img2 = imgs

    seam = optimal_seam(img1, img2)
    img3 = blend_images(img1, img2, seam)
    img3 = crop(img3)

    seam_x, seam_y = zip(*seam)

    # Showing the results
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    
    plt.axis('off')
    plt.title("Image 2")
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.subplot(1, 3, 2)

    plt.axis('off')
    plt.title("Image 1 Transformed")
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))

    plt.subplot(1, 3, 3)
    plt.axis('off')
    plt.title("Image 3")
    plt.plot(seam_y, seam_x, 'r-', linewidth=1)
    plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
    plt.show()

    cv2.imwrite('./results/stitched_5.jpg', img3)


# Task 6 - SuperGlue stitching
def stitch_two_sg():
    img1 = cv2.imread('./undistorted/img1.png')
    img2 = cv2.imread('./undistorted/img2.png')

    pairs = './matching/img1_img2_matches.npz'
    npz = np.load(pairs)

    kp0 = npz['keypoints0']
    kp1 = npz['keypoints1']
    matches = npz['matches']
    confidence = npz['match_confidence']

    valid = matches > -1
    points_src = kp0[valid]
    points_dst = kp1[matches[valid]]

    H = find_transform(points_src, points_dst)
    imgs = project_canvas(img1, img2, H)

    imgs = color_correction(imgs)
    img1, img2 = imgs

    seam = optimal_seam(img1, img2)
    img3 = blend_images(img1, img2, seam)
    img3 = crop(img3)

    seam_x, seam_y = zip(*seam)

    # Showing the results
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    
    plt.axis('off')
    plt.title("Image 2")
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.subplot(1, 3, 2)

    plt.axis('off')
    plt.title("Image 1 Transformed")
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))

    plt.subplot(1, 3, 3)
    plt.axis('off')
    plt.title("Image 3")
    plt.plot(seam_y, seam_x, 'r-', linewidth=1)
    plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
    plt.show()

    cv2.imwrite('./results/stitched_6.jpg', img3)


# Task 7 - SuperGlue stitching
def stitch_five():
    right = cv2.imread('./undistorted/img1.png')
    for i in range(1, 5):
        left = cv2.imread(f'./undistorted/img{i+1}.png')
        pairs = np.load(f'./matching/img{i}_img{i+1}_matches.npz')

        valid = pairs['matches'] > -1
        points_src = pairs['keypoints0'][valid]
        points_dst = pairs['keypoints1'][pairs['matches'][valid]]

        H = find_transform(points_src, points_dst)
        imgs = project_canvas(right, left, H)

        right, left = color_correction(imgs)

        seam = optimal_seam(right, left)
        stitched = blend_images(right, left, seam)
        stitched = crop(stitched)

        cv2.imshow('stitched', stitched)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        right = stitched.copy()
    
    cv2.imwrite('./results/stitched_7.jpg', stitched)



# Task 1 - Camera calibration.
# calibrate_and_compare()

# Task 2 - Image projection.
# test_proj()

# Task 3 - Find transformation matrix.
# test_transform()

# Task 4 - Manual image transformation.
# manual_transform()

# Task 5 - Stitching images.
# run_project_canvas()

# Task 6 - SuperGlue stitching
# stitch_two_sg()

# Task 7 - SuperGlue stitching
# stitch_five()