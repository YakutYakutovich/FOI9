from Recognition import RecognizePlate
import cv2
import time


# Инициализация объекта для распознавания номера
recog = RecognizePlate()
recog.load_detection_model('models/best.pt')
recog.load_recognize_model('')


# Обычно виртуальная камера OBS находится на индексе 0 или 1
class Camera:
    def __init__(self):
        self.__camera_index = 1
        self.__cap = cv2.VideoCapture(self.__camera_index)

    def __check_opened(self):
        if not self.__cap.isOpened():
            print("Не удалось открыть видеопоток.")
            exit()

    def __get_frame(self):
        ret, frame = self.__cap.read()
        return frame.copy()

    def see(self):
        self.__check_opened()

        while True:
            # Получаем кадр с камеры
            frame = self.__get_frame()

            # Распознаем номер на кадре
            print(recog.recognize(frame))

            # Отображаем кадр в окне
            # cv2.imshow("Camera Feed", frame)
            # time.sleep(3)
            # cv2.destroyWindow("Camera Feed")

            # Ждем 1 секунду и проверяем нажатие клавиши для выхода
            # if cv2.waitKey(1) & 0xFF == ord('q'):  # Если нажать 'q', программа завершится
            #     break

            # Ждем 1 секунду
            time.sleep(1)

        # Закрываем окно после завершения работы
        self.__cap.release()
        cv2.destroyAllWindows()

# Запуск камеры
Camera().see()
