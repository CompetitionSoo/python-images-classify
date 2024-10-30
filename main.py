'''
이미지 학습을 마친 모델을 읽어들여서
opencv 에서 읽어들인 배열을 예측시키기.
'''

from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np

# 출력 형식을 지수 형식으로 하지 않음
# 지수 = 1e 알파벳으로 된 숫자들 
np.set_printoptions(suppress=True)

# 모델 읽어들이기
model = load_model("keras_model.h5", compile=False)    # 학습을 다하였기 대문에 가저오기만 하면된다.

# 모델 라벨 읽기
class_names = open("labels.txt", "r").readlines()     # 한줄당 하나씩 가져온다 코드를
# bottle
# Phone

camera = cv2.VideoCapture(0)

while True:
    # 프레임 읽기
    ret, image = camera.read()

    # 예측을 위한 이미지 축소
    frame = image[:, :, :]                                                # 이미지의 내용을 frame에 저장하겠다
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)   # 표준모델로 학습하겠다 하고 시작을눌렀음


    # 예측 가능한 형태로 변형
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # 효과적인 예측을 위한 정규화
    image = (image / 127.5) - 1

    # 예측하기
    prediction = model.predict(image)   
    index = np.argmax(prediction)              # 가장큰 인덱스를 구할수있다.
    class_name = class_names[index]            # 가장큰 인덱스명 = 확률
    confidence_score = prediction[0][index]

    # 확률 출력하기 
    # print("Class:", class_name[2:], end="")
    # print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")
    cv2.putText(frame, f"{class_name[2:-1]} : {str(np.round(confidence_score * 100))[:-2]}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2) 
    # putText 이미지에 확률 추출하기
    
    # 이미지 보기
    cv2.imshow("Webcam Image", frame)

    # 키 입력 기다리기
    keyboard_input = cv2.waitKey(1)

    # 27은 ESC를 의미함
    if keyboard_input == 27:
        break

# 카메라와 창 종료
camera.release()        # 카메라 리소스 해제
cv2.destroyAllWindows() # 모든 OpenCV 창 닫기