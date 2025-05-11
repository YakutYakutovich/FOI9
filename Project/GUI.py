import time

import cv2
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk
from Recognition import RecognizePlate


class VideoCaptureApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.__recog = RecognizePlate()
        self.__recog.load_detection_model('best.pt')
        self.__recog.load_recognize_model('')

        # Открыть видеопоток с камеры
        # Раскомментировать тот вариант который надо
        # Для того чтобы работала встроенная камера нужно в self.cap заменить cv2.VideoCapture(1) на cv2.VideoCapture(0)
        self.cap = cv2.VideoCapture(1)
        #self.cap = cv2.VideoCapture("rtsp://admin:12345@192.168.1.100:554/Streaming/Channels/101")
        if not self.cap.isOpened():
            print("Не удалось открыть камеру.")
            exit()

        # Создание метки для отображения видео
        self.label = Label(window)
        self.label.pack()

        # Запуск обновления видео
        self.update()

        # Завершение работы
        self.window.mainloop()

    def update(self):
        # Чтение кадра из камеры
        ret, frame = self.cap.read()
        if ret:
            # Преобразование кадра в формат, совместимый с Tkinter
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            self.__recog.recognize(frame)

            image = Image.fromarray(frame)
            photo = ImageTk.PhotoImage(image=image)

            # Обновление изображения в метке
            self.label.config(image=photo)
            self.label.image = photo

        time.sleep(0.5)

        # Повторный вызов update для следующего кадра
        self.window.after(10, self.update)

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()

root = tk.Tk()
app = VideoCaptureApp(root, "Видео с камеры")
