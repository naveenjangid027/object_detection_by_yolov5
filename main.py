import cv2
import numpy as np
import pickle
import histMaker
import time
import histVisual

# Open the video capture
cap = cv2.VideoCapture(0)  # Use 0 for webcam or replace with video file path

# Create and initialize the color histogram
hist = None
roi = None

def load_histogram(filename='histogram.pkl'):
    with open(filename, 'rb') as file:
        hist = pickle.load(file)
    print(f"Histogram loaded from {filename}")
    return np.array(hist)

def bounding_box(x,y,w,h,t):
    '''This function returns the new, expanded bounding box'''
    nx = max(0, x - t)
    ny = max(0, y - t)
    max_x = min(1920, x + w + t)
    max_y = min(1080, y + h + t)

    return int(nx), int(ny), int(max_x), int(max_y)

hist_queue = []

def update_histogram(new_hist):
    hist_queue.append(new_hist)
    if len(hist_queue) > 10:  # Keep a maximum of 10 histograms in the queue
        hist_queue.pop(0)
    
    weights = [1 / (i + 1) for i in range(len(hist_queue))]  # Assign weights to each histogram
    total = sum(weights)
    normalized_weights = [weight / total for weight in weights]
    weighted_avg_hist = np.zeros_like(new_hist)
    for i, hist in enumerate(hist_queue):
        weighted_avg_hist += hist * normalized_weights[i]
    weighted_avg_hist *= 0.5
    weighted_avg_hist += load_histogram('orange.pkl') * 0.5
    return weighted_avg_hist

def start():
    hist_load_time = time.time()
    hist = load_histogram('orange.pkl')
    hist_queue.append(hist)
    visual = histVisual.HistogramVisualizer()
    while True:
        current_time = time.time()
        if current_time - hist_load_time > 0.1:  # Load histogram every 1 second
            hist_load_time = current_time
        
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert the frame to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Calculate the back projection
        dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)
        
        # Apply a binary threshold to eliminate low probability pixels
        _, thresh = cv2.threshold(dst, 50, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours and find the largest one (assumed to be the ball)
        max_area = 0
        max_contour = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                max_contour = contour
        
        if max_contour is not None:
            # Draw the largest contour
            cv2.drawContours(frame, [max_contour], 0, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(max_contour)
            x, y, max_x, max_y = bounding_box(x,y,w,h, 100)

            roi = frame[y:max_y, x:max_x]  # extract ROI from original thresholded image
            cv2.rectangle(frame,(x,y),(max_x, max_y),(0, 0, 255),3)
            if roi.size > 0:  # Check if ROI is not empty
                #Create new histogram
                hist = update_histogram(histMaker.create_histogram(roi))
                print("new histogram saved")
                visual.animate(hist)
            else:
                print("ROI is empty")

        # Display the original frame and the thresholded image
        cv2.imshow('Frame', frame)
        cv2.imshow('Thresholded', thresh)
        
        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close the windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start()
