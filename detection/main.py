from ultralytics import YOLO
import requests
import numpy as np
import cv2

model = YOLO('yolov8n.pt')

# Foto URLsi
url = 'https://images.unsplash.com/photo-1438761681033-6461ffad8d80?q=80&w=1000&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MTl8fGh1bWFufGVufDB8fDB8fHww'

# Fotoyu infir
response = requests.get(url)
if response.status_code != 200:
    raise Exception(f"Error fetching image from URL: {url}")

# indirilen fotoyu numoy arrayine çevirip OpenCV ile decode
image_array = np.array(bytearray(response.content), dtype=np.uint8)
image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)


results = model.predict(image, save=False, conf=0.25)

frame_with_boxes = image.copy()

# Sadece insanları algıla
person_class_id = 0
for det in results[0].boxes:

    x1, y1, x2, y2, conf, cls_id = map(int, [*det.xyxy[0], det.conf[0], det.cls[0]])

    # Eğer insansa detect
    if cls_id == person_class_id:
        cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2) # Yeşil kutuya al
        cv2.putText(frame_with_boxes, f'Person {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),
                    2)

# Ekran boyutuna göre manuel olarak fotoyu yeniden boyutlandır
screen_resolution = (900, 700)
if frame_with_boxes.shape[1] > screen_resolution[0] or frame_with_boxes.shape[0] > screen_resolution[1]:
    frame_with_boxes = cv2.resize(frame_with_boxes, screen_resolution, interpolation=cv2.INTER_AREA)

# imshow ile çalıştır
cv2.imshow('Detected Image', frame_with_boxes)

# Herhangi bir tuşla programı kapat
cv2.waitKey(0)
cv2.destroyAllWindows()