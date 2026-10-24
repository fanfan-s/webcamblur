import cv2
import urllib.request
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# lol
model_path = 'hand_landmarker.task'
if not os.path.exists(model_path):
    print("donwnload hand landmarker")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, model_path)
    print("seles")

# mediapipe tasks
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO, 
    num_hands=2
)
detector = vision.HandLandmarker.create_from_options(options)

# nyalain kamera
cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
print("lgi nyalain webcam")

# variabel waktu video
frame_index = 0 

while True:
    sukses, frame = cap.read()
    if not sukses:
        print("buka kamera ulang")
        break
        
    # ini buat mirror kamera ny
    frame = cv2.flip(frame, 1)
    
    # convert mediapipe format jir
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # timestamp buat frame
    timestamp_ms = frame_index * 33
    frame_index += 1
    
    # deteksi tangan 
    hasil = detector.detect_for_video(mp_image, timestamp_ms)
    
    pose_2_jari = False
    
    # kalo tangan kedetect
    if hasil.hand_landmarks:
        for hand_landmarks in hasil.hand_landmarks:
            lm = hand_landmarks
            
            # logika buat deteksi pose ny
            telunjuk_naik = lm[8].y < lm[6].y
            tengah_naik = lm[12].y < lm[10].y
            manis_turun = lm[16].y > lm[14].y
            kelingking_turun = lm[20].y > lm[18].y
            
            if telunjuk_naik and tengah_naik and manis_turun and kelingking_turun:
                pose_2_jari = True
                
    # efek blur
    if pose_2_jari:
        frame = cv2.GaussianBlur(frame, (99, 99), 0)
        
    # Tampilkan ke layar
    cv2.imshow("ini webcam, pencet q buat klr", frame)
    
    # break webcam ny
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
# buat clear cache kamera
cap.release()
cv2.destroyAllWindows()