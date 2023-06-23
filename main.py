#Uses caffe 
import numpy as np
import cv2 as cv

#Set to True if input is video, False if input is image
video = True

#weights and architectures for face, age, and gender detection models
faceProto = "training/deploy.prototxt"
faceModel = "training/res10_300x300_ssd_iter_140000_fp16.caffemodel"

genderModel = "training/deploy_gender.prototxt"
genderProto = "training/gender_net.caffemodel"

ageModel = "training/deploy_age.prototxt"
ageProto = "training/age_net.caffemodel"

#mean values of BGR that are subtracted from images to handle illumination changes
means = (78.4263377603, 87.7689143744, 114.895847746)

#load face Caffe and prediction models
faceNet = cv.dnn.readNetFromCaffe(faceProto, faceModel)
genderNet = cv.dnn.readNetFromCaffe(genderModel, genderProto)
ageNet = cv.dnn.readNetFromCaffe(ageModel, ageProto)
genderList = ['Male', 'Female']
ageList = ['(0 - 2)', '(4 - 6)', '(8 - 12)', '(15 - 20)', '(25 - 32)', '(38 - 43)', '(48 - 53)', '(60 - 100)']

#Detects faces in images
def getFaceBox(frame, conf_threshold=0.5):
    blob = cv.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123]) #converting image to blob
    faceNet.setInput(blob) #inputting blob
    output = np.squeeze(faceNet.forward())  #getting output and using numpy to strip away unnecessary outerlayers of the array
    faces = [] #result list
     # Loop over the faces detected
    for i in range(output.shape[0]):
        confidence = output[i, 2]
        if confidence > conf_threshold:
            box = output[i, 3:7] * \
                np.array([frame.shape[1], frame.shape[0],
                         frame.shape[1], frame.shape[0]])
            # convert to integers
            start_x, start_y, end_x, end_y = box.astype(int)
            # widen the box a little
            start_x, start_y, end_x, end_y = start_x - \
                10, start_y - 10, end_x + 10, end_y + 10
            start_x = 0 if start_x < 0 else start_x
            start_y = 0 if start_y < 0 else start_y
            end_x = 0 if end_x < 0 else end_x
            end_y = 0 if end_y < 0 else end_y
            # append to our list
            faces.append((start_x, start_y, end_x, end_y))
    return faces

#Displays frames
def displayframe(title,frame):
    cv.imshow(title,frame)
    cv.waitKey(0)
    cv.destroyAllWindows()

#Returns gender predictions from image
def genderPrediction(face):
    blob = cv.dnn.blobFromImage(face, 1, (227, 227), means, False, False)
    print("starting gender")
    genderNet.setInput(blob)  
    print("finishing gender")
    return genderNet.forward()

#Returns age predictions from image
def agePrediction(face):
    blob = cv.dnn.blobFromImage(face, 1, (227, 227), means, False, False)
    print("starting age")
    ageNet.setInput(blob)    
    print("finish age")

    return ageNet.forward()


def ageandGenderPredictionForImage(imagep):
    image = cv.imread(imagep) #read image
    frame = image.copy() #copy it
    faces = getFaceBox(frame) #find face images

    for i,(x,y,z,h) in enumerate(faces): #loop through face images

        face_img = frame[y:h, x:z]  #extract just face

        ages = agePrediction(face_img) #input to prediction model for age
        genders = genderPrediction(face_img) #^ for gender

        i = genders[0].argmax(0) #index of gender with highest confidence
        gender = genderList[i] #gender of highest confidence
        genderConfidence = genders[0][i] #gender confidence

        i = ages[0].argmax(0) #^ but with ages
        age = ageList[i]
        ageConfidence = ages[0][i]

        #Draw box
        cv.rectangle(frame, (x,y), (z,h), (0, 255,0), 2)

        #Draw label
        ylabel = y -15 #make label a little higher
        while ylabel < 15: ylabel += 15

        label = f"{gender}-{genderConfidence*100:.1f}%, {age}-{ageConfidence*100:.1f}%" 
        print(label)
        cv.putText(frame, label, (x,ylabel), cv.FONT_HERSHEY_SIMPLEX, .54,(0, 255,0), 2)

    cv.imshow("Age and Gender Guesser", frame)
    cv.waitKey(0)
    cv.destroyAllWindows()

def ageandGenderPredictionForVideo():
    video = cv.VideoCapture(0) #0 means default computer camera

    while True:
        _, img = video.read()
        frame = img.copy() #copy it
        faces = getFaceBox(frame) #find face images

        for i,(x,y,z,h) in enumerate(faces): #loop through face images

            face_img = frame[y:h, x:z]  #extract just face

            ages = agePrediction(face_img) #input to prediction model for age
            genders = genderPrediction(face_img) #^ for gender

            i = genders[0].argmax(0) #index of gender with highest confidence
            gender = genderList[i] #gender of highest confidence
            genderConfidence = genders[0][i] #gender confidence

            i = ages[0].argmax(0) #^ but with ages
            age = ageList[i]
            ageConfidence = ages[0][i]

            #Draw box
            cv.rectangle(frame, (x,y), (z,h), (0, 255,0), 2)

            #Draw label
            ylabel = y -15 #make label a little higher
            while ylabel < 15: ylabel += 15

            label = f"{gender}-{genderConfidence*100:.1f}%, {age}-{ageConfidence*100:.1f}%" 
            print(label)
            cv.putText(frame, label, (x,ylabel), cv.FONT_HERSHEY_SIMPLEX, .54,(0, 255,0), 2)

        cv.imshow("Age and Gender Guesser", frame)
        if cv.waitKey(1) == ord("q"):
            break        
    cv.destroyAllWindows()


if not video:
    ageandGenderPredictionForImage("oldladyface.webp")
else: 
    ageandGenderPredictionForVideo()
