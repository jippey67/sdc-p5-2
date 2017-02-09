from lesson_functions import *
from sklearn.externals import joblib
from moviepy.editor import VideoFileClip
from collections import deque
from scipy.ndimage.measurements import label

class determinator():
    # this class determines whether a
    def __init__(self):
        self.model = joblib.load('model1.pkl')
        self.X_scaler = joblib.load('scaler1.pkl')

    def scale_and_predict(self, features):
        scaled_X = self.X_scaler.transform(features)
        pred = self.model.predict(scaled_X)
        return pred

    def extract_features(self, img, spatial_size=(32, 32),
                         hist_bins=32, orient=9,
                         pix_per_cell=8, cell_per_block=2, hog_channel=0,
                         spatial_feat=True, hist_feat=True, hog_feat=True):

        features = []
        file_features = []
        feature_image = img

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
            # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:, :, channel],
                                                         orient, pix_per_cell, cell_per_block,
                                                         vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                                pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
        #print(color_space,spatial_feat, hist_feat, hog_feat,spatial_size,hist_bins,orient,pix_per_cell,cell_per_block,hog_channel,features)
        # Return list of feature vectors
        return features

def slide_window( x_start_stop=[None, None], y_start_stop=[None, None],
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size

    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_windows = np.int(xspan / nx_pix_per_step) - 1
    ny_windows = np.int(yspan / ny_pix_per_step) - 1
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs * nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys * ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]

            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

def process_image(src_image):
    image = cv2.cvtColor(src_image, cv2.COLOR_RGB2YCrCb)
    # parameters for sliding windows
    all_search = [[[320, 960], [400, 496], [64, 64], [0.75, 0.75]], [[320, 960], [400, 544], [96, 96], [0.75, 0.75]],
                  [[0, 1280], [400, 592], [128, 128], [0.75, 0.75]]]
    on_windows = []
    for i in range(len(all_search)):
        #search all windows and add positions of found cars to on_windows
        slides = slide_window(x_start_stop=all_search[i][0], y_start_stop=all_search[i][1], xy_window=all_search[i][2],
                              xy_overlap=all_search[i][3])
        for slide in slides:
            img = cv2.resize(image[slide[0][1]:slide[1][1], slide[0][0]:slide[1][0], :], (64, 64))
            feat = det.extract_features(img)

            pred = det.scale_and_predict(feat)
            if pred > 0.5:
                on_windows.append(slide)
    box_img = draw_boxes(src_image, on_windows)
    # create a heat sheet for the found assumed cars
    heat_sheet = np.zeros((720,1280))
    for win in on_windows:
        heat_sheet[win[0][1]:win[1][1], win[0][0]:win[1][0]] += 1
    heat_sheet[heat_sheet<=2]=0 # threshold for individual sheet
    heat_queue.append(heat_sheet) # add heatsheet to queue
    heat_map = np.zeros((720,1280))
    for i in range(len(heat_queue)):
        heat_map += heat_queue[i]
    heat_map[heat_map <= 4] = 0 # threshold for queue
    imago = np.zeros((720,1280,3))
    #imago [:,:,1] = 30 * heat_map
    labels = label(heat_map)
    imago = draw_labeled_bboxes(np.copy(src_image), labels)
    return imago


det = determinator() # object for car recognition
heat_queue = deque(np.zeros((720,1280)),20)  #queue for keeping consecutive heat sheets

clip = VideoFileClip("project_video.mp4")
augmented_clip = clip.fl_image(process_image)
augmented_clip.write_videofile('augmented_video.mp4', audio=False)