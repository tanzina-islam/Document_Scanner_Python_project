import cv2
import numpy as np
import utlis

########################################################################
pathImage = "1.jpg"

webCamFeed = False
cap = cv2.VideoCapture(1)
cap.set(10, 160)
heightImg = 640
widthImg = 480
########################################################################

utlis.initializeTrackbars()
count = 0

while True:

    if webCamFeed:success, img = cap.read()
    else:img = cv2.imread(pathImage)

    #Resize Image
    img = cv2.resize(img, (widthImg, heightImg))

    imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)  # Creat a blank image for testing
    #img = cv2.imread(pathImage)
    #resized_image = cv2.resize(img, (widthImg, heightImg))# Resize image

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)# Convert image to gary scale
    blur_image = cv2.GaussianBlur(gray_image, (5, 5), 1)  # Add Gaussian blur
    thres = utlis.valTrackbars()  # Get the track bur value for thresholds
    threshold_image = cv2.Canny(blur_image, thres[0], thres[1]) # Apply canny blur
    kernel = np.ones((5, 5))
    image_Dial = cv2.dilate(threshold_image, kernel, iterations=2)  # Apply dilation
    threshold_image = cv2.erode(image_Dial, kernel, iterations=1) # Apply erosion

    # Find all contours
    imgContour = img.copy() # Copy image for display purpose
    imgBigContour = img.copy() # Copy image for display purpose
    contours, hierarchy = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   # Find all contours

    # Find the biggest contours
    biggest, maxArea = utlis.biggestContour(contours)  # Find the biggest contours
    #print(biggest)
    if biggest.size != 0:
        biggest = utlis.reorder(biggest)
        cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 20)  # Draw the biggest contours
        imgBigContour = utlis.drawRectangle(imgBigContour, biggest, 2)
        pts1 = np.float32(biggest)  # Prepare for point for wrap
        pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])  # Prepare for point for wrap
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

        # Remove 20 pixel form each side
        imgWarpColored = imgWarpColored[20:imgWarpColored.shape[0] - 20, 20:imgWarpColored.shape[1] - 20]
        imgWarpColored = cv2.resize(imgWarpColored, (widthImg, heightImg))

        # Apply adaptive threshold
        imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
        imgAdaptiveThre = cv2.adaptiveThreshold(imgWarpGray, 255, 1, 1, 7, 2)
        imgAdaptiveThre = cv2.bitwise_not(imgAdaptiveThre)
        imgAdaptiveThre = cv2.medianBlur(imgAdaptiveThre, 3)

        # Image array for display
        imageArray = ([img, gray_image, threshold_image, imgContour],
                      [imgBigContour, imgWarpColored, imgWarpGray, imgAdaptiveThre])

    else:
        imageArray = ([img, gray_image ,threshold_image, imgContour],
                      [imgBlank, imgBlank, imgBlank, imgBlank])

    # Labels for display
    lables = [["Original", "Gray", "Threshold", "Contours"],
              ["Biggest Contour", "Warp Prespective", "Warp Gray", "Adaptive Threshold"]]

    stackedImage = utlis.stackImages(imageArray, 0.75, lables)#.stackImages(imageArray, 0.75, lables)
    cv2.imshow("Result", stackedImage)

    # Save image when 's' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("Scanned/myImage" + str(count) + ".jpg", imgWarpColored)
        cv2.rectangle(stackedImage, ((int(stackedImage.shape[1] / 2) - 230), int(stackedImage.shape[0] / 2) + 10),
                      (1000, 250), (0, 255, 0), cv2.FILLED)
        cv2.putText(stackedImage, "Scan Saved", (int(stackedImage.shape[1] / 2) - 200, int(stackedImage.shape[0] / 2)),
                    cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255), 5, cv2.LINE_AA)
        cv2.imshow('Result', stackedImage)
        cv2.waitKey(300)
        count += 1
