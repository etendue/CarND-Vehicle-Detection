# # Advanced Lane Finding
# 
# The Project
# ---
# 
# The goals / steps of this project are the following:
# 
# * Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
# * Apply a distortion correction to raw images.
# * Use color transforms, gradients, etc., to create a thresholded binary image.
# * Apply a perspective transform to rectify binary image ("birds-eye view").
# * Detect lane pixels and fit to find the lane boundary.
# * Determine the curvature of the lane and vehicle position with respect to center.
# * Warp the detected lane boundaries back onto the original image.
# * Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
# 


import numpy as np
import cv2
import glob


# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None


def calibrate_camera(cal_file_path):
    # use glob to match file with pattern, here the pattern is file extension '.jpg'
    calfiles = glob.glob(cal_file_path + "/*.jpg")
    # Arrays to store object pongs and image points from all the images
    # the above image has 6x 9 corners
    nr = 6
    nc = 9
    # prepare object points, smart idea to use np.mgrid
    objp = np.zeros((nr * nc, 3), np.float32)
    objp[:, :2] = np.mgrid[:nc, :nr].T.reshape(-1, 2)

    objpoints = []  # 3D points with Z axis always 0, Why?
    imgpoints = []  # 2D points on image, value in pixels

    for file in calfiles:
        img = cv2.imread(file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # be careful of the 2nd parameter, patternSize=(columns,rows)
        ret, corners = cv2.findChessboardCorners(gray, (nc, nr), None)

        # if found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
        else:
            print("corners not found in file : {}".format(file))
    # calibrate camera to get transformation matrix

    _, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, (img.shape[1], img.shape[0]), None, None)
    return mtx, dist


# ## 3. Use color transforms, gradients, etc., to create a thresholded binary image.     

# ### 3.1 implement a gradient threshold filter on x or y axis


# the image must be in format BGR or BGRA e.g. read by cv2.imread
# since we convert grayscale use pixel format COLOR_BGR2GRAY
def abs_xy_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    """ apply absolute value thresh hold on a image in x or y direction
    argugments:
    img -- input image
    sobel_kernel -- the sobel kernel size (default 3)
    orient -- orientation, default(x axis)
    thresh -- threshold for low and high magnitude(default 0,255)
    return -- a binary image with gradient within threshold 
    
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    sobel = np.abs(sobel)
    binary_output = np.zeros_like(sobel)

    binary_output[(sobel >= thresh[0]) & (sobel <= thresh[1])] = 1

    return binary_output


# the image must be in format BGR or BGRA e.g. read by cv2.imread
# since we convert grayscale use pixel format COLOR_BGR2GRAY
def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):
    """ apply magnitude thresh hold on a image
    argugments:
    img -- input image
    sobel_kernel -- the sobel kernel size (default 3)
    thresh -- threshold for low and high magnitude
    return -- a binary image with gradient within threshold 
    
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    sobel = np.sqrt(sobelx ** 2 + sobely ** 2)
    sobel_scale = np.uint8(255 * sobel / np.max(sobel))
    binary_output = np.zeros_like(sobel_scale)
    binary_output[(sobel_scale >= thresh[0]) & (sobel_scale <= thresh[1])] = 1

    return binary_output


# the image must be in format BGR or BGRA e.g. read by cv2.imread
# since we convert grayscale use pixel format COLOR_BGR2GRAY
def dir_thresh(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
    """ apply angle threshold on a image
    argugments:
    img -- input image
    sobel_kernel -- the sobel kernel size (default 3)
    thresh -- threshold for angle to horizon, i.e. horizontal line has angle=0
    return -- a binary image with gradient within angle threshold 
    
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    sobel_angle = np.arctan2(np.abs(sobely), np.abs(sobelx))

    binary_output = np.zeros_like(sobel_angle)
    binary_output[(sobel_angle >= thresh[0]) & (sobel_angle <= thresh[1])] = 1

    return binary_output


# the image must be in format BGR or BGRA e.g. read by cv2.imread
# since we convert grayscale use pixel format COLOR_BGR2GRAY
def hls_saturation_thresh(img, thresh=(0, 255)):
    """ apply saturation threshold on a image
    argugments:
    img -- input image
    thresh -- threshold for saturation channel in hls color space
    return -- a binary image with saturation within angle threshold 
    
    """
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    s = hls[:, :, 2]

    binary_output = np.zeros_like(s)
    binary_output[(s >= thresh[0]) & (s <= thresh[1])] = 1

    return binary_output


# ## 4. Apply a perspective transform to rectify binary image ("birds-eye view") 

def get_perspective_matrix():
    # get matrix of perspective transformation
    # source points from image
    src = np.float32([[247, 691], [601, 447], [679, 447], [1061, 691]])
    # target points rectangle
    dst = np.float32([[250, 720], [250, 0], [950, 0], [950, 720]])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M,Minv


# ## 5. put all together with gradient and color filtering


def process_img_with_filters(img, x_fil=None, y_fil=None, m_fil=None, a_fil=None, s_fil=None):
    """
    Apply various gradient and color filter to get a binary image
    
    Arguments:
    img -- input image
    x_fil -- abs(x) axis sobel filter parameter, (ksize, low_threshold, high_threshold)
    y_fil -- abs(y) axis sobel filter parameter, (ksize, low_threshold, high_threshold)
    m_fil -- abs(magnitude) sobel filter parameter,(ksize, low_threshold, high_threshold)
    a_fil -- abs(angle in radius) sobel filter parameter,(ksize, low_threshold, high_threshold)
    s_fil -- hls color space s channel filter, (low_threshold, high_threshold)
    return -- a binary image filtered by different options.
    """
    # create binary image for output
    img_bin = np.zeros_like(img[:, :, 0])
    absx_bin = img_bin.copy()
    absy_bin = img_bin.copy()
    absm_bin = img_bin.copy()
    dir_bin = img_bin.copy()
    s_bin = img_bin.copy()

    if x_fil != None:
        absx_bin = abs_xy_thresh(img, orient='x', sobel_kernel=x_fil[0], thresh=(x_fil[1], x_fil[2]))
    if y_fil != None:
        absy_bin = abs_xy_thresh(img, orient='y', sobel_kernel=y_fil[0], thresh=(y_fil[1], y_fil[2]))
    if m_fil != None:
        absm_bin = mag_thresh(img, sobel_kernel=m_fil[0], thresh=(m_fil[1], m_fil[2]))
    if a_fil != None:
        dir_bin = dir_thresh(img, sobel_kernel=a_fil[0], thresh=(a_fil[1], a_fil[2]))
    if s_fil != None:
        s_bin = hls_saturation_thresh(img, thresh=s_fil)

    # combine the filters
    # keep both x and y gradients within threshold 
    img_bin[((absx_bin == 1) & (absy_bin == 1))] = 1
    # keep magnitude and direction within threshold 
    img_bin[((absm_bin == 1) & (dir_bin == 1))] = 1
    # keep satuation within threshold
    img_bin[s_bin == 1] = 1

    return img_bin


# ## 6. Detect lane pixels and fit to find the lane boundary.

# use window sliding to search the lane from binary image
def detect_lanes(bin_img, plot=False):
    """
    use window sliding to search the lane from binary image
    Arguments:
    bin_img -- a binary image for lane searching
    plot -- flag whether to return a image with search area and polyfit lines
    return -- left lane and right lane (left,right,out_img) if plot==False,out_img=None
    """

    img_height = bin_img.shape[0]
    img_width = bin_img.shape[1]
    # middle point x coordinate
    mid_px = img_height // 2
    # middle point y coordinate
    mid_py = img_width // 2

    # Take a histogram of the bottom half of the image
    histogram = np.sum(bin_img[mid_py:, :], axis=0)

    leftx_base = np.argmax(histogram[:mid_px])
    rightx_base = np.argmax(histogram[mid_px:]) + mid_px

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(img_height / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = bin_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # records the rectangle window positions
    rectangles = []
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img_height - (window + 1) * window_height
        win_y_high = img_height - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        rectangles.append([win_xleft_low, win_y_low, win_xleft_high, win_y_high,
                           win_xright_low, win_y_low, win_xright_high, win_y_high])

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    ploty = np.linspace(0, img_height - 1, img_height)
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # assign the values to Line() ojects
    left_lane = Line()
    right_lane = Line()
    left_lane.detected = True
    right_lane.detected = True
    left_lane.current_fit = left_fit
    right_lane.current_fit = right_fit
    left_lane.recent_xfitted = left_fitx
    right_lane.recent_xfitted = right_fitx

    out_img = None
    if plot == True:
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((bin_img, bin_img, bin_img)) * 255
        # some system will see values as signed  so convert to unsigned byte, 
        out_img = out_img.astype(np.uint8)

        # mark the lane pixels to be red on left lane and blue on right lane
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        for x1, y1, x2, y2, x3, y3, x4, y4 in rectangles:
            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(out_img, (x3, y3), (x4, y4), (0, 255, 0), 2)

        left_line_pts = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        right_line_pts = np.array([np.transpose(np.vstack([right_fitx, ploty]))])

        cv2.polylines(out_img, np.int_([left_line_pts]), False, (255, 255, 0), 2)
        cv2.polylines(out_img, np.int_([right_line_pts]), False, (255, 255, 0), 2)

    return left_lane, right_lane, out_img


# from the next frame of video
# It's now much easier to find line pixels!
def search_lanes(bin_img, previous_lanes=None, plot=False):
    """
    search the lanes based on previous detected lines
    arguments:
    bin_img -- binary image as input
    previous_lanes -- a tuple of (left,right) Lines to confine the line search area
                     if its value is None, then detect_lanes() is called
    plot -- flag whether to return a image with search area and polyfit lines
    return -- left,right,out_img ( out_img = None when plot== False)
    """

    img_height = bin_img.shape[0]
    img_width = bin_img.shape[1]

    if previous_lanes == None:
        return detect_lanes(bin_img)

    nonzero = bin_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100

    # get the polyfit parameters from previous image
    left_fit = previous_lanes[0].current_fit
    right_fit = previous_lanes[1].current_fit

    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) &
                      (nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) &
                       (nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, img_height - 1, img_height)
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # assign the values to Line() ojects
    left_lane = Line()
    right_lane = Line()
    left_lane.current_fit = left_fit
    right_lane.current_fit = right_fit
    left_lane.recent_xfitted = left_fitx
    right_lane.recent_xfitted = right_fitx

    out_img = None

    if plot == True:
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((bin_img, bin_img, bin_img)) * 255
        out_img = out_img.astype(np.uint8)
        window_img = np.zeros_like(out_img, dtype=np.uint8)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # draw the polyfit lines
        left_line_pts = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        right_line_pts = np.array([np.transpose(np.vstack([right_fitx, ploty]))])

        cv2.polylines(out_img, np.int_([left_line_pts]), False, (255, 255, 0), 2)
        cv2.polylines(out_img, np.int_([right_line_pts]), False, (255, 255, 0), 2)

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        out_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    return left_lane, right_lane, out_img


# ## 7. Determine the curvature of the lane and vehicle position with respect to center.

def calculate_geometries(img, left_lane, right_lane):
    """
    calculate the curvature, line_base_pos = vehicle center to lane
    arguments:
    img -- input image, used to get size of image
    left_lane -- the detected left line
    right_lane -- the detected right line
    return -- None
    """
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
    img_height = img.shape[0]
    img_width = img.shape[1]

    ploty = np.linspace(0, img_height - 1, img_height)
    left_fitx = left_lane.recent_xfitted
    right_fitx = right_lane.recent_xfitted
    # the y axis value at the bottom
    y_eval = np.max(ploty)
    # here calculate physical curvature
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty * ym_per_pix, left_fitx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, right_fitx * xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix +
                           left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix +
                            right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])

    left_lane.radius_of_curvature = left_curverad
    right_lane.radius_of_curvature = right_curverad
    # calculate line_base_pos
    left_lane.line_base_pos = (img_width / 2 - left_fitx[-1]) * xm_per_pix
    right_lane.line_base_pos = (right_fitx[-1] - img_width / 2) * xm_per_pix

    return


def sanity_check(left_lane, right_lane):
    """
    check if the lanes detected is confident
    arguments:
    left_lane -- the detected left line
    right_lane -- the detected right line
    return -- True if lanes are OK, else False
    """
    # Define conversions in x and y from pixels space to meters
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    # 1. Checking that they have similar curvature
    # set 10% deviation
    dev = 2 * (left_lane.radius_of_curvature - right_lane.radius_of_curvature) / (left_lane.radius_of_curvature +
                                                                                  right_lane.radius_of_curvature)
    if np.abs(dev) > 0.2:
        err = "curvature is {:.02f}% deviation".format(dev * 100)
        return False, err
    # 2. Checking that they are separated by approximately the right distance horizontally
    distances = (right_lane.recent_xfitted - left_lane.recent_xfitted) * xm_per_pix

    # the distance should between 2.8m and 5m
    if (np.min(distances) < 2.8) | (np.max(distances) > 5):
        err = "two lanes have min distance {},max distance {}".format(np.min(distances),
                                                                      np.max(distances))
        return False, err

    # 3. Checking that they are roughly parallel
    # deviation within 0.5m 
    dev = np.abs(distances - np.mean(distances))
    if np.max(dev) > 0.5:
        err = "deviation from average is {}".format(np.max(dev))
        return False, err

    return True, ""


# ## 8. Warp the detected lane boundaries back onto the original image.  


def warp_lanes_back(img, left_lane, right_lane, mtx):
    """
    warp the detected line from bird eye back to normal view angle 
    arguments:
    img -- input image
    left_lane -- the detected left line
    right_lane -- the detected right line
    mtx -- perspective transform matrix
    return -- a image with lane area marked
    """
    img_height = img.shape[0]
    out_img = np.zeros_like(img, dtype=np.uint8)

    if left_lane.bestx != None:
        left_fitx = left_lane.bestx
    else:
        left_fitx = left_lane.recent_xfitted
    if right_lane.bestx != None:
        right_fitx = right_lane.bestx
    else:
        right_fitx = right_lane.recent_xfitted

    ploty = ploty = np.linspace(0, img_height - 1, img_height)

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(out_img, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    out_img = cv2.warpPerspective(out_img, mtx, (img.shape[1], img.shape[0]))

    # output curvature and offset to lane
    text1 = "Radius of Curvature ={:.01f}m".format(
        (left_lane.radius_of_curvature + right_lane.radius_of_curvature) / 2)

    offset = right_lane.line_base_pos - left_lane.line_base_pos
    pos = "left" if offset > 0 else "right"
    text2 = "Vehicle is {:.01f}m {} of center".format(
        np.abs(offset), pos)

    cv2.putText(out_img, text1, (1, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(out_img, text2, (1, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return out_img


# ## 9. Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

# ### 9.1 the pipeline to process a image 


# the last good detected lines
last_good_lanes = None
bad_frame_count = 0
bad_frame_threshold = 2
mtx,dist = calibrate_camera("camera_cal")
M, Minv = get_perspective_matrix()


def pipeline(img):
    """
    the pipeline to process the image through following steps
    - Apply a distortion correction to raw images.
    - Use color transforms, gradients, etc., to create a thresholded binary image.
    - Apply a perspective transform to rectify binary image ("birds-eye view").
    - Detect lane pixels and fit to find the lane boundary.
    - Determine the curvature of the lane and vehicle position with respect to center.
    - Warp the detected lane boundaries back onto the original image.
    """
    global last_good_lanes
    global bad_frame_count
    global bad_frame_threshold
    global mtx,dist,M, Minv

    img_size = (img.shape[1], img.shape[0])

    # apply distortin correction
    img_undistort = cv2.undistort(img, mtx, dist, None, mtx)
    # perspective transformation
    img_trans = cv2.warpPerspective(img_undistort, M, img_size, flags=cv2.INTER_LINEAR)
    # Use color, grandients filter to create a thresholded binary image
    img_bin = process_img_with_filters(img_trans, x_fil=(3, 10, 255), y_fil=(3, 10, 255), m_fil=(3, 20, 255),
                                       a_fil=(15, 0.5, np.pi / 2), s_fil=(120, 255))
    # first detect the lanes
    if last_good_lanes == None or (bad_frame_count >= bad_frame_threshold):
        left_lane, right_lane, _ = detect_lanes(img_bin)
    else:
        # find the lanes using Look-Ahead Filter if previous lanes are found
        left_lane, right_lane, _ = search_lanes(img_bin, previous_lanes=last_good_lanes)

    # calucate the curvature and line_base_pos
    calculate_geometries(img, left_lane, right_lane)

    res, _ = sanity_check(left_lane, right_lane)
    if res == True:
        bad_frame_count = 0
        if last_good_lanes != None:
            # average across two frames
            left_lane.bestx = (last_good_lanes[0].recent_xfitted + left_lane.recent_xfitted) / 2
            right_lane.bestx = (last_good_lanes[1].recent_xfitted + right_lane.recent_xfitted) / 2
        last_good_lanes = (left_lane, right_lane)
    else:
        bad_frame_count += 1

    if last_good_lanes != None:
        out_img = warp_lanes_back(img, last_good_lanes[0], last_good_lanes[1], Minv)
        return cv2.addWeighted(img_undistort, 1, out_img, 0.3, 0)
    else:
        return img_undistort