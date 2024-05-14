import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, QMessageBox
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
import requests
import pickle
import os
import pandas as pd
import numpy as np
import uuid


from ultralytics import YOLO
import requests
import numpy as np
import cv2

# Load the YOLO model
model = YOLO('aiModels\yolov8n.pt')
def detect_human_in_profile_pic(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Error fetching image from URL: {url}")

    # Convert the image content to a NumPy array
    image_array = np.array(bytearray(response.content), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    # Predict using the model
    results = model.predict(image, save=False, conf=0.25)

    # Check for person class (class_id for 'person' is usually 0)
    person_class_id = 0
    person_detected = False
    for det in results[0].boxes:
        cls_id = int(det.cls[0])
        if cls_id == person_class_id:
            person_detected = True
            # Draw bounding box
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Person"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if person_detected:
        # Generate a unique filename and save to the labeledImages directory
            directory = 'labeledImages'
            if not os.path.exists(directory):
                os.makedirs(directory)
            unique_filename = os.path.join(directory, f"labeled_image_{uuid.uuid4().hex}.jpg")
            cv2.imwrite(unique_filename, image)
            return unique_filename
        else:
            return None

def calculate_ratio(username): 
    num_digits = sum(char.isdigit() for char in username)
    total_length = len(username)
    return num_digits / total_length if total_length > 0 else 0

def check_url(url):
    default_pic_patterns = ["44884218_345707102882519_2446069589734326272_n.jpg"]
    return 0 if any(pattern in url for pattern in default_pic_patterns) else 1

def count_wordsin_fullname(fullname:str):
    return len(fullname.split(' '))


class SecondWindow(QWidget):
    closed = pyqtSignal() 
    
    def __init__(self, username: str, url: str, path:str, parameters: list, predict: list):
        super().__init__()  
         
        self.username = username
        self.PPurl = url
        self.parameters = parameters
        self.imagePath = path
       
        self.predict = predict
        self.initUI()
    def closeEvent(self, event):
        self.closed.emit() 
        super().closeEvent(event)
        
    
    def initUI(self):
        self.layout = QVBoxLayout()
        self.layout.setAlignment(Qt.AlignCenter)  

        if self.imagePath:
            self.label_image_path = QLabel()
            self.label_image_path.setPixmap(QPixmap(self.imagePath))
            self.label_image_path.setStyleSheet("""
                QLabel {
                    color: white;
                    border: 3px solid white;
                    border-radius: 15px;
                    margin-top: 6px;
                    margin-bottom: 6px;
                    
                }
            """)  
            self.label_image_path.setAlignment(Qt.AlignCenter)  
            
            self.layout.addWidget(self.label_image_path)
            
        image = QImage()
        image.loadFromData(requests.get(self.PPurl).content)
        self.setStyleSheet("background-color: black;")  
        self.image_label = QLabel(self)
        self.image_label.setPixmap(QPixmap(image))
        self.image_label.setAlignment(Qt.AlignCenter)  
        self.image_label.setStyleSheet("""
            QLabel {
                color: white;
                border: 3px solid white;
                border-radius: 15px;
                margin-top: 6px;
                margin-bottom: 6px;
            }
        """)  
        self.layout.addWidget(self.image_label)
        
        info_labels = [
            ('Kullanıcı adı:', self.username),
            ('Takipçi:', self.parameters[1]),
            ('Takip Ettikleri:', self.parameters[2]),
            ('Post:', self.parameters[0])
        ]
        for label, data in info_labels:
            info_label = QLabel(f"{label} {data}", self)
            info_label.setAlignment(Qt.AlignCenter)
            info_label.setStyleSheet("color: white; font-size: 16px; font-weight: bold;")
            self.layout.addWidget(info_label)

        
        model_labels = ['YoloV8','VotingClassifier Model','StackingClassifier Model', 'Decision Tree', 'Extra Trees', 'Gradient Boosting', 'Random Forest','Ada Boost']
    
        for model, prediction in zip(model_labels, self.predict):
            label = QLabel(f"{model} Tahmini: {prediction}", self)
            label.setAlignment(Qt.AlignCenter)  
            label.setStyleSheet("color: black; font-size: 16px; background-color: white; padding 5px;") 
            self.layout.addWidget(label)

        self.setLayout(self.layout)
        self.setWindowTitle('Sahte mi Gerçek mi Analiz')
        self.setGeometry(400, 400, 550, 550)


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.secondWindow = None
        self.postCount = 0
        self.followers = 0
        self.follows = 0
        self.HumanInPP = 0
        self.initUI()

    def initUI(self):
        self.setStyleSheet("background-color: black;")  
        self.main_layout = QVBoxLayout(self)
        self.center_layout = QHBoxLayout()
        self.title_label = QLabel("Sahte mi? Gerçek mi?", self)
        self.title_label.setStyleSheet(
        "QLabel {"
        "   color: white;"
        "   font-size: 20px;"  
        "   font-weight: bold;"  
        "}"
    )
        self.title_label.setAlignment(Qt.AlignCenter) 
        self.main_layout.addWidget(self.title_label)   
        self.username_input = QLineEdit(self)
        self.username_input.setPlaceholderText("Lütfen kullanıcı adını giriniz")
        self.username_input.setStyleSheet(
            "QLineEdit {"
            "   background-color: white;"
            "   border: 2px solid red;"
            "   border-radius: 5px;"  
            "   padding: 5px;"  
            "}"
        )
        self.center_layout.addWidget(self.username_input)

        self.call_api_button = QPushButton("Ara", self)
        self.call_api_button.setStyleSheet(
            "QPushButton {"
            "   background-color: white;"
            "   color: black;"
            "   border-radius: 10px;"
            "   padding: 10px 20px;" 
            "}"
        )
        self.call_api_button.clicked.connect(self.on_call_api)
        self.center_layout.addWidget(self.call_api_button)

        self.main_layout.addLayout(self.center_layout)
        self.setWindowTitle('Instagram Fake Detector')
        self.setGeometry(300, 300, 350, 250)

    def on_call_api(self):
        self.username = self.username_input.text()
        attributes = self.call_api(self.username)
        print(attributes)
        if attributes:
            
            self.HumanInPP = detect_human_in_profile_pic(attributes[0])
            data_df = pd.DataFrame([attributes[1:]])
            predicts = self.get_predictions(data_df)
            self.show_results(self.username, attributes[0], predicts)

    def get_predictions(self, data_df):
        predicts = []
        if not self.HumanInPP:
            predicts.append("Fake")
        else:
            predicts.append("Gerçek")
            
        for filename in os.listdir("aiModels"):
            if filename.endswith('.pkl'):
                with open(os.path.join("aiModels", filename), 'rb') as file:
                    model = pickle.load(file)
                    prediction = model.predict(data_df)
                    predicts.append("Gerçek" if prediction[0] == 0 else "Fake")
        return predicts

    def show_results(self, username, url, predicts):
            
        if self.secondWindow is None:  
            self.secondWindow = SecondWindow(username, url,self.HumanInPP, [self.postCount,self.followers,self.follows], predicts)
            self.secondWindow.closed.connect(self.secondWindowClosed) 
            self.secondWindow.show()

    def call_api(self, username):
        try:
            
            url = "https://instagram-scraper-api2.p.rapidapi.com/v1/info"
            querystring = {"username_or_id_or_url":username}
            headers = {
            "X-RapidAPI-Key": "30cd363e79msh521ad370e078889p16b943jsnb0e88b1054ed",
            "X-RapidAPI-Host": "instagram-scraper-api2.p.rapidapi.com"
        }
            response = requests.get(url, headers=headers, params=querystring).json()
            return self.parse_response(response, username)
        except Exception as e:
            print(e)
            QMessageBox.warning(self, "Hata", "Kullanıcı bulunamadı", QMessageBox.Ok)
            return []

    def parse_response(self, response, username):
        self.followers = response['data']['follower_count']
        self.follows = response['data']['following_count']
        self.postCount = response['data']['media_count']
        aktivite_oranı = np.round(self.postCount / self.followers, 2) if self.followers != 0 else 0
        fullName = response['data']['full_name']
        isEqual = int(fullName == username)
        isPrivate = int(response['data']['is_private'])
        externalUrl = 0 if response['data']['external_url'] else 1

        attributes = [
            response['data']['profile_pic_url'],
            check_url(response['data']['profile_pic_url']),
            calculate_ratio(username),
            count_wordsin_fullname(fullName),
            calculate_ratio(fullName),
            isEqual,
            len(response['data']['biography_with_entities']['raw_text']),
            externalUrl,
            isPrivate,
            self.postCount,
            self.followers,
            self.follows,
            aktivite_oranı,
            1 if self.followers > self.follows else 0
        ]
        return attributes
    
    def secondWindowClosed(self):
        self.secondWindow = None

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())
