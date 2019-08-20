import cv2

font = cv2.FONT_HERSHEY_SIMPLEX
cap = cv2.VideoCapture(0)
# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
while True:
    # Read the input image
    ret, frame = cap.read()
    height, width, channels = frame.shape
    # Convert into grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 8)
    # Draw rectangle around the faces
    if len(faces)>0:
        cx=0
        cy=0
        msg=""
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cx=cx+x+w/2
            cy=cy+y+h/2

            print(w*h)
            if w*h>9000:
                msg=msg+"hatra, "
            if w*h<4000:
                msg=msg+"elore, "

        cx=cx/len(faces)
        cy=cy/len(faces)

        if cx < width*0.4:
            msg=msg+"jobbra, "
        if cx > width*0.6:
            msg=msg+"balra, "
        if cy < height*0.4:
            msg=msg+"fel, "
        if cy > height*0.6:
            msg=msg+"le"
        cv2.rectangle(frame, (int(cx)-2, int(cy)-2), (int(cx)+4, int(cy)+4), (0, 255, 0), 4)


        
        cv2.putText(frame,msg,(10,height-15),font,1,(0,0,255),2)

    # Display the output
    cv2.imshow('img', frame)

    c = cv2.waitKey(1)

    if c == 27:
        break

cap.release()
# finally, close the window
cv2.destroyAllWindows()