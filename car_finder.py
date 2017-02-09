import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from lesson_functions import * #get_hog_features, bin_spatial, color_hist #, slide_window, draw_boxes
import pandas as pd
import time

# from skimage.feature import hog
# from skimage import color, exposure
# images are divided up into vehicles and non-vehicles

cars_fl = glob.glob('./vehicles/**/*.png', recursive=True)
notcars_fl = glob.glob('./non-vehicles/**/*.png', recursive=True)

def trainer(c_space,spatial_size=(64,64),hist_bins=64,sf=True,hf=True,hg=True,orient=9,
                     pix_per_cell=8, cell_per_block=2, hog_channel="ALL"):
    car_features = extract_features(c_space, cars_fl,spatial_size=spatial_size,hist_bins=hist_bins,spatial_feat=sf,hist_feat=hf,hog_feat=hg,orient=orient,
                     pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)
    notcar_features = extract_features(c_space, notcars_fl,spatial_size=spatial_size,hist_bins=hist_bins,spatial_feat=sf,hist_feat=hf,hog_feat=hg,orient=orient,
                     pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    len_feat = len(X)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X

    scaled_X = X_scaler.transform(X)
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)
    svc = LinearSVC()
    # Train the SVC
    svc.fit(X_train, y_train)
    score = svc.score(X_test, y_test)
    print('Test Accuracy of SVC = ', score)
    return [c_space,sf,hf,hg,str(spatial_size[0])+"x"+str(spatial_size[1]),hist_bins,orient,pix_per_cell,cell_per_block,hog_channel ,score, len_feat], svc, X_scaler

def round0():
    color_space = "HSV"
    scores = []

    #HOG_chs = [0,1,2,"ALL"]
    HOG_chs = ["ALL"]
    orient_bins = [5,9,13]
    pix_per_cell = [4,8,16]
    cell_per_block = [1,2]
    st = time.time()
    for hog in HOG_chs:
        for orient in orient_bins:
            for pix in pix_per_cell:
                for cell in cell_per_block:
                    print(hog, orient, pix, cell)
                    t2=time.time()
                    print(round(t2-st,2),' seconds passed since start')
                    accuracy = trainer(c_space=color_space, sf=False, hf=False, hg=True,orient=orient,pix_per_cell=pix,cell_per_block=cell,hog_channel=hog)
                    scores.append(accuracy)
    df = pd.DataFrame(np.array(scores))
    df.to_csv("round0.csv")


def round1():
    for i in range(10):
        color_spaces = ["HSV","YCrCb","LUV","HLS","YUV"]
        scores = []
        for space in color_spaces:
            for method in range(7):
                if method == 0:
                    A = False
                    B = False
                    C = True
                if method == 1:
                    A = False
                    B = True
                    C = False
                if method == 2:
                    A = False
                    B = True
                    C = True
                if method == 3:
                    A = True
                    B = False
                    C = False
                if method == 4:
                    A = True
                    B = False
                    C = True
                if method == 5:
                    A = True
                    B = True
                    C = False
                if method == 6:
                    A = True
                    B = True
                    C = True
                print(str(i),space, method)
                accuracy = trainer(space,sf=A,hf=B,hg=C)
                scores.append(accuracy)

        df = pd.DataFrame(np.array(scores))
        df.to_csv(str(i)+"round1.csv")

def round2():
    for i in range(10):
        scores=[]
        color_spaces = "YCrCb"
        spatial_size = [(16,16),(32,32),(64,64)]
        histo_bins = [32,64]
        A = False
        B = True
        C = False
        for size in spatial_size:
            for bins in histo_bins:
                accuracy = trainer(color_spaces, spatial_size=size, hist_bins=bins,sf=A,hf=B,hg=C)
                scores.append(accuracy)
        print('round:',i)
        df = pd.DataFrame(np.array(scores))
        df.to_csv(str(i)+"round2histonly.csv")

def round3():
    from sklearn.externals import joblib

    accuracy, model, Xscaler = trainer(c_space="YCrCb", spatial_size=(32,32 ), hist_bins=32, sf=True, hf=True, hg=True, hog_channel=0)
    print (accuracy)
    joblib.dump(model, 'model3.pkl')
    joblib.dump(Xscaler, 'scaler3.pkl')

'''
run round0 for HOG SPACE analysis
run round1 for combined feature analysis
run round2 for analysis within one color space
run round3 for training a model and saving it together with the feature scaler
'''

#round0()
#round1()
#round2()
round3()

