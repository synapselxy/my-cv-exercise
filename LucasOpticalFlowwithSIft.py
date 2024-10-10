# Snippet to calculate Optical Flow between two frames

# Calculate Optical Flow between the two frames
import cv2
import numpy as np
import matplotlib.pyplot as plt

def imshow(*args, **kwargs):
    cv2.imshow(*args, **kwargs)

# Read two frames
frame1 = cv2.imread( "assets/frame0.png")
frame2 = cv2.imread( "assets/frame1.png")
# plt.imshow(frame1)
# plt.imshow(frame2)

# Convert to grayscale
frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

# Parameters for lucas kanade optical flow
win_size = 25
lk_params = dict( winSize = (win_size, win_size),
                  maxLevel = 0,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                              10, 0.01))

# Find good points to track
points = []
# Initialize SIFT detector
sift = cv2.SIFT_create()

# # Find keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(frame1, None)
# kp2, des2 = sift.detectAndCompute(frame2, None)
points = np.array([kp.pt for kp in kp1], dtype=np.float32)

# Calculate Optical Flow using Lucas-Kanade
flow, status, errors = cv2.calcOpticalFlowPyrLK(frame1_gray, frame2_gray, points, None, **lk_params)

# Show on image
for i in range(len(flow)):
    a, b = flow[i].ravel()
    c, d = points[i].ravel()
    #  Convert to int
    a, b, c, d = int(a), int(b), int(c), int(d)
    # Draw line between points
    frame2 = cv2.line(frame2, (a, b), (c, d), (0, 255, 0), 1)
    # Draw circle at point
    frame2 = cv2.circle(frame2, (a, b), 1, (0, 0, 255), -1)

imshow("frame2", frame2)

cv2.waitKey(0)
cv2.destroyAllWindows()