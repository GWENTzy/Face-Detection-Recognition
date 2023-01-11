import cv2 as cv
import os 
import numpy as np
import math
from matplotlib import pyplot as plt

# this function will return a list of all the paths in the current directory
def get_path_list(root_path):
    '''
        To get a list of path directories from root path

        Parameters
        ----------
        root_path : str
            Location of root directory
        
        Returns
        -------
        list
            List containing the names of the sub-directories in the
            root directory
    '''
    
    # compute list directories in root path
    root = os.listdir(root_path)
    
    # initilize array to store folder path in root directory 
    folder_path_list = [] 

    # looping through root directory
    for index, folder_path in enumerate(root):
        # append folder path
        folder_path_list.append(folder_path)

    return folder_path_list

# this function is designed to get the class ID of a given file
def get_class_id(root_path, train_names):
    '''
        To get a list of train images and a list of image classes id

        Parameters
        ----------
        root_path : str
            Location of images root directory
        train_names : list
            List containing the names of the train sub-directories
        Returns
        -------
        list
            List containing all image in the train directories
        list
            List containing all image classes id
    '''
    
    # initilize array to store image and class id
    image_list = []
    image_class_list = []

    # looping through list of folder or train names path
    for image_class, folder_path in enumerate(train_names):
        full_folder_path = f'{root_path}/{folder_path}'
        
        # compute list image path
        folder = os.listdir(full_folder_path)
        
        # looping through list of image in folder
        for index, image_path in enumerate(folder):
            full_image_path =  f'{full_folder_path}/{image_path}'
            
            # load the image
            image = cv.imread(full_image_path, 0)
            
            # append image list with the image
            image_list.append(image)
            
            # append image class list with class of the image
            image_class_list.append(image_class)
        
    return image_list, image_class_list

# this function takes in an image list and an image classes list, and returns the number of faces detected in each image
def detect_faces_and_filter(image_list, image_classes_list=None):
    '''
        To detect a face from given image list and filter it if the face on
        the given image is less than one

        Parameters
        ----------
        image_list : list
            List containing all loaded images
        image_classes_list : list, optional
            List containing all image classes id
        
        Returns
        -------
        list
            List containing all filtered and cropped face images in grayscale
        list
            List containing all filtered faces location saved in rectangle
        list
            List containing all filtered image classes id
    ''' 
    
    # initilize array to store image and class face, classes, and filtered faces in rectangle
    face_list = []
    face_rectangle_list = []
    face_class_list = []

    # initiate cascade classifier
    classifier = cv.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    
    # check if there is image_classes_list
    # if its there then its for train
    if image_classes_list is not None:
        
        # looping through list of the image and the classes
        for image, image_class in zip(image_list, image_classes_list):
            
            # computer detected faces with the classifier
            detected_faces = classifier.detectMultiScale(
                image,
                scaleFactor = 1.3,
                minNeighbors = 3
            )
            
            # check if there is detected faces
            # if none then continue
            if(len(detected_faces) == 0):
                continue
            
            # looping through detected faces
            for x, y, w, h in detected_faces:
                # crop the image
                face_image = image[y:y+w, x:x+h]
                
                # append face image to face list
                face_list.append(face_image)
                
                # append face class list with class of the image
                face_class_list.append(image_class)
        
        return face_list, face_rectangle_list, face_class_list

    # if image_classes_list is none then its for test
    for image in image_list:
        
        # load the image
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        
        # computer detected faces with the classifier
        detected_faces = classifier.detectMultiScale(
            image,
            scaleFactor = 1.3,
            minNeighbors = 3
        )
        
        # check if there is detected faces
        # if none then continue
        
        if(len(detected_faces) < 1):
            continue
        
        # looping through detected faces    
        for x, y, w, h in detected_faces:
            # crop the image
            face_image = image[y:y+w, x:x+h]
            # append face image to face list
            face_list.append(face_image)
            # append face rectangle list with the faces location
            face_rectangle_list.append([x,y,w,h])

    return face_list, face_rectangle_list, face_class_list

# this function is used to train a classifier to recognize faces in images
def train(train_face_grays, image_classes_list):
    '''
        To create and train face recognizer object

        Parameters
        ----------
        train_face_grays : list
            List containing all filtered and cropped face images in grayscale
        image_classes_list : list
            List containing all filtered image classes id
        
        Returns
        -------
        object
            Recognizer object after being trained with cropped face images
    '''
    
    # initiate face detector model
    face_recognizer = cv.face.LBPHFaceRecognizer_create()
    
    # fit the model on training data
    face_recognizer.train(
        train_face_grays,
        np.array(image_classes_list)
    )

    return face_recognizer

# this function is used to get the images data from a test directory
def get_test_images_data(test_root_path):
    '''
        To load a list of test images from given path list

        Parameters
        ----------
        test_root_path : str
            Location of images root directory
        
        Returns
        -------
        list
            List containing all loaded gray test images
    '''
    
    # compute list images in test root path
    test_root = os.listdir(test_root_path)
    
    # initilize array to store test images
    image_list = []
    
    # looping through test_root directory
    for image_path in (test_root):
        full_image_path = f'{test_root_path}/{image_path}'
        # load the image
        image = cv.imread(full_image_path)
        # append image list with the image
        image_list.append(image)

    return image_list

# this function is used to predict the probability of a given face being in the test_faces_gray list
def predict(recognizer, test_faces_gray):
    '''
        To predict the test image with the recognizer

        Parameters
        ----------
        recognizer : object
            Recognizer object after being trained with cropped face images
        train_face_grays : list
            List containing all filtered and cropped face images in grayscale

        Returns
        -------
        list
            List containing all prediction results from given test faces
    '''
    
    # initilize array to store prediction results
    prediction_result_list = []
    
    # looping through test faces
    for face in test_faces_gray:    
        # predict
        prediction, confidence = recognizer.predict(face)
       
        # append prediction_result_list with the prediction results
        prediction_result_list.append(prediction)
        
    return prediction_result_list

# this function is used to draw the predicted results of a given image in the context of a list of faces
def draw_prediction_results(predict_results, test_image_list, test_faces_rects, train_names):
    '''
        To draw prediction results on the given test images and acceptance status

        Parameters
        ----------
        predict_results : list
            List containing all prediction results from given test faces
        test_image_list : list
            List containing all loaded test images
        test_faces_rects : list
            List containing all filtered faces location saved in rectangle
        train_names : list
            List containing the names of the train sub-directories

        Returns
        -------
        list
            List containing all test images after being drawn with
            final result
    '''
    
    # initilize array to store images final result
    image_result_list = []
    
    # looping through the prediction and test images
    for prediction, face_image, face_rectangel in zip(predict_results, test_image_list, test_faces_rects):
        x, y, w, h = face_rectangel
        
        # draw a rectangle as the face location within face image
        face_image = cv.rectangle(
            face_image,
            (x,y), 
            (x+w, y+h), 
            (0, 255, 0), 
            1
        )
        
        # check if prediction classes is youtuber or twitch streamer
        # then add youtuber or twitch in text prediction
        if train_names[prediction] in ['Pewdiepie', 'Jacksepticeye']:
            text = f'{train_names[prediction]} - Youtube'
        else:
            text = f'{train_names[prediction]} - Twitch'
        
        # put prediction text above the rectangle
        face_image = cv.putText(face_image, text, (x, y-10), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        
        # append image_result_list with the face image result
        image_result_list.append(face_image)

    return image_result_list

# this function takes an image list as input and returns the result of combining all the images in the list
def combine_and_show_result(image_list):
    '''
        To show the final image that already combine into one image

        Parameters
        ----------
        image_list : nparray
            Array containing image data
    '''
    
    #creating a figure with a size of 10x7 pixels
    plt.figure(figsize=(10,7))
    
    for index, image in enumerate(image_list):
        
        #for each index in the list, it creates a subplot on the left side of the figure with size 2x3 
        plt.subplot(2, 3, index+1)
        
        #it draws an image using cv.cvtColor()
        #then adds x-axis ticks to show where they are located on screen
        plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        plt.xticks([])
        plt.yticks([])

    #after drawing all four images at once onto screen, it calls tight_layout() to arrange them nicely before showing them off with show()
    plt.tight_layout()
    plt.show()


'''
You may modify the code below if it's marked between

-------------------
Modifiable
-------------------

and

-------------------
End of modifiable
-------------------
'''
if __name__ == "__main__":

    '''
        Please modify train_root_path value according to the location of
        your data train root directory

        -------------------
        Modifiable
        -------------------
    '''
    train_root_path = "dataset/train"
    '''
        -------------------
        End of modifiable
        -------------------
    '''

    train_names = get_path_list(train_root_path) #labels_list
    train_image_list, image_classes_list = get_class_id(train_root_path, train_names) #faces, indexes
    train_face_grays, _, filtered_classes_list = detect_faces_and_filter(train_image_list, image_classes_list)
    recognizer = train(train_face_grays, filtered_classes_list)

    '''
        Please modify train_root_path value according to the location of
        your data train root directory

        -------------------
        Modifiable
        -------------------
    '''
    test_root_path = "dataset/test"
    '''
        -------------------
        End of modifiable
        -------------------
    '''

    test_image_list = get_test_images_data(test_root_path)
    test_faces_gray, test_faces_rects, _ = detect_faces_and_filter(test_image_list)
    predict_results = predict(recognizer, test_faces_gray)
    predicted_test_image_list = draw_prediction_results(predict_results, test_image_list, test_faces_rects, train_names)
    
    combine_and_show_result(predicted_test_image_list)