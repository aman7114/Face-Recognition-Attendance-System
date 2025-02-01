import cv2
import face_recognition
import numpy as np
import datetime
import os

# Function to mark attendance (tracks attendance per session)
def mark_attendance(name, attendance_record):
    if name not in attendance_record:
        with open("attendance.csv", "a") as f:
            today = datetime.date.today()
            now = datetime.datetime.now().strftime("%H:%M:%S")
            f.write(f"{today},{now},{name}\n")
        attendance_record.append(name)

# Load images and encode faces (using face_recognition for simplicity)
def load_images_and_encode(tolerance=0.4):  # Adjust tolerance as needed
    known_faces_encodings = []
    known_faces_names = []
# DataSet
    image_paths = [
        "photos\_Aman Sharma.jpg",
        "photos\_Aman Sharma (2).jpg",
        "photos\_Aman Sharma (3).jpg",
        "photos\_Aman Sharma (4).jpg",
        "photos\_Aman Sharma (5).jpg",
        "photos\Ratnesh Singh Vishen.jpg",
        "photos\Ratnesh Singh Vishen (2).jpg",
        "photos\Ratnesh Singh Vishen (3).jpg",
        "photos\Ratnesh Singh Vishen (4).jpg",
        "photos\Rachit Kumar.jpg",
        "photos\Rachit Kumar (1).jpg",
        "photos\Rachit Kumar (2).jpg",
        "photos\Rachit Kumar (3).jpg",
        "photos\Dikshit Guleria.png",
        "photos\Dikshit Guleria  (2).png",
        "photos\Dikshit Guleria  (3).png",
        "photos\Dikshit Guleria  (4).png",
        "photos\Vansh Phalswal.jpg",
        "photos\Vansh Phalswal (2).jpg",
        
        # Add more paths as needed
    ]

    for path in image_paths:
        image = face_recognition.load_image_file(path)
        face_encodings = face_recognition.face_encodings(image)
        if face_encodings:
            face_encoding = face_encodings[0]  # Take the first face encoding
            known_faces_encodings.append(face_encoding)
            known_faces_names.append(os.path.splitext(os.path.basename(path))[0])
        else:
            print(f"No face detected in {path}")

    return known_faces_encodings, known_faces_names


# Main function for face recognition
def recognize_faces(cloud_api_key=None, cloud_api_endpoint=None):
    # Open the video capture device (check camera index if needed)
    video_capture = cv2.VideoCapture(0)

    # Check if camera opened successfully
    if not video_capture.isOpened():
        print("Error opening video capture device. Check your camera connection or permissions.")
        exit()  # Exit the program if camera fails to open

    attendance_record = []  # List to track attendance per session

    # Load pre-encoded faces with adjustable tolerance
    known_faces_encodings, known_faces_names = load_images_and_encode()

    while True:
        ret, frame = video_capture.read()

        if not ret:
            print("Error reading frame from camera.")
            break  # Exit the loop if frame reading fails

        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        # Check if any faces were detected before iterating
        if face_locations:
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # Use minimum number of True matches for confident recognition (adjust threshold)
                matches = face_recognition.compare_faces(known_faces_encodings, face_encoding, tolerance=0.5)  # Adjust tolerance as needed
                name = "Unknown"

                if sum(matches) >= 2:  # Require at least 2 matches for confident recognition
                    first_match_index = matches.index(True)
                    name = known_faces_names[first_match_index]

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (0, 255, 0), 1)

                if name != "Unknown":
                    mark_attendance(name, attendance_record)
        else:
            print("No faces detected in this frame.")

        cv2.imshow("Video", frame)

        # Exit loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release camera resources
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_faces()
