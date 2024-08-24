import cv2

# Load the Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

ip_vid = 'data/people1.mp4'
capture = cv2.VideoCapture(ip_vid)

# Check if the video capture is opened successfully
if not capture.isOpened():
    print(f"Error: Unable to open video file {ip_vid}")
    exit()

# Get video properties
frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = capture.get(cv2.CAP_PROP_FPS)

print(f"Video properties - Width: {frame_width}, Height: {frame_height}, FPS: {fps}")

if fps == 0:
    print("Error: FPS value is zero. Setting FPS to 30.")
    fps = 30

# Output video file path
op_vid = 'output/blurred1.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(op_vid, fourcc, fps, (frame_width, frame_height))

if not out.isOpened():
    print(f"Error: Unable to open video writer for file {op_vid}")
    capture.release()
    exit()

frame_count = 0

while capture.isOpened():
    ret, frame = capture.read()
    
    if not ret:
        print(f"End of video or error reading frame at frame {frame_count}.")
        break
    
    try:
        # print(f"Processing frame {frame_count}")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            blur_face = cv2.GaussianBlur(face_roi, (99, 99), 30)
            frame[y:y+h, x:x+w] = blur_face
        
        out.write(frame)
        frame_count += 1
    except Exception as e:
        print(f"Error processing frame {frame_count}: {e}")
        continue

capture.release()
out.release()
cv2.destroyAllWindows()