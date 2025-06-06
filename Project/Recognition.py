import easyocr
import ultralytics
from ultralytics import YOLO
import cv2
import re
from alignment import straighten_plate_from_image


"""
tesseract

Установить easyocr и ultralytics
pip install easyocr ultralytics
"""


class RecognizePlate:
    """
    Класс для распознавания автомобильных номеров на изображении с использованием моделей
    YOLO для обнаружения номерных знаков и EasyOCR для их распознавания.
    """

    def __init__(self):
        """
        Инициализирует объект RecognizePlate с пустыми моделями для обнаружения
        и распознавания текста, а также указывает устройство выполнения ('cpu').
        """
        self.__model_detection = None
        self.__model_recognize = None
        self.device = 'cpu'

        self.__alf_rus = 'АВЕКНМОРСТУХ'
        self.__alf_eng = 'ABEKHMOPCTYX'
        self.__alfall = self.__alf_eng + self.__alf_rus
        self.__nums = '0123456789ОOоo'
        self.__dict_rus2eng = {}
        for i in range(len(self.__alf_eng)):
            self.__dict_rus2eng[self.__alf_rus[i]] = self.__alf_eng[i]

    def load_detection_model(self, path_2_model: str) -> None:
        """
        Загружает модель YOLO для обнаружения номерных знаков на изображении.

        :param path_2_model: Путь к файлу модели YOLO
        """
        self.__model_detection = YOLO(path_2_model)

    @staticmethod
    def check_string_on_mask(string: str, mask: list) -> str | None:
        for i in range(len(string) - len(mask)):
            counter = 0
            mask_i = 0
            for c in range(i, i + len(mask)):
                if string[c] in mask[mask_i]:
                    counter += 1
                mask_i += 1
            if counter == len(mask):
                return string[i: i + len(mask)]

    def match_pattern_plate(self, string: str) -> str | None:
        nums = self.__nums
        alf_eng = self.__alf_eng
        alf_rus = self.__alf_rus
        alf = alf_rus.lower() + alf_rus + alf_eng.lower() + alf_eng
        mask1 = [alf, nums, nums, nums, alf, alf, nums, nums, nums]
        mask2 = [alf, nums, nums, nums, alf, alf, nums, nums]

        check1 = self.check_string_on_mask(string, mask1)
        check2 = self.check_string_on_mask(string, mask2)

        print(f'Check all plate:', end=' ')

        if check1 is None and check2 is None:
            return None
        elif check1 is None and not (check2 is None):
            print(check2, end='  |  ')
            return check2
        else:
            print(check1, end='  |  ')
            return check1

    def match_pattern_number(self, string: str) -> str | None:
        """
        Проверяет, соответствует ли распознанный текст шаблону автомобильного номера.

        :param string: Распознанный текст
        :return: bool — True, если текст соответствует шаблону, иначе False
        """
        nums = self.__nums
        alf_eng = self.__alf_eng
        alf_rus = self.__alf_rus
        alf = alf_rus.lower() + alf_rus + alf_eng.lower() + alf_eng
        mask = [alf, nums, nums, nums, alf, alf]

        for i in range(len(string) - 6):
            counter = 0
            mask_i = 0
            for c in range(i, i + 6):
                if string[c] in mask[mask_i]:
                    counter += 1
                mask_i += 1
            if counter == 6:
                return string[i: i + 6]

    def match_pattern_region(self, string: str) -> str | None:
        """
                Проверяет, соответствует ли распознанный текст шаблону региона автомобильного номера.

                :param string: Распознанный текст
                :return: bool — True, если текст соответствует шаблону, иначе False
                """
        nums = self.__nums
        mask1 = [nums, nums, nums]
        mask2 = [nums, nums]

        len_mask1, len_mask2 = len(mask1), len(mask2)

        for i in range(len(string) - len_mask1):
            counter = 0
            mask_i = 0
            for c in range(i, i + len_mask1):
                if string[c] in mask1[mask_i]:
                    counter += 1
                mask_i += 1
            if counter == len_mask1:
                return string[i: i + len_mask1]

        for i in range(len(string) - len_mask2):
            counter = 0
            mask_i = 0
            for c in range(i, i + len_mask2):
                if string[c] in mask2[mask_i]:
                    counter += 1
                mask_i += 1
            if counter == len_mask2:
                return string[i: i + len_mask2]

    def load_recognize_model(self, path_2_model: str) -> None:
        """
        Загружает модель EasyOCR для распознавания текста на номерных знаках.

        :param path_2_model: Путь к модели EasyOCR (необязательно)
        """
        self.__model_recognize = easyocr.Reader(['ru'])

    def recognize(self, image: 'cv2.imread') -> str:
        """
        Основной метод для распознавания номера на изображении.

        :param image: Изображение, считанное с помощью cv2.imread
        :return: строка с распознанным номером или None, если номер не распознан
        """

        detection_results = self.__detection_plate(image)

        best_result = None
        best_score = 0

        for box in detection_results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            cropped = image[y1:y2, x1:x2]  # Обрезаем изображение по координатам

            # Применяем выравнивание к обрезанному изображению
            aligned_images = straighten_plate_from_image(cropped)

            for aligned in aligned_images:
                ocr_results = self.__model_recognize.readtext(aligned)

                for (bbox, text, prob) in ocr_results:
                    text_cleaned = self.clean_string(text.replace(" ", "").upper())

                    pattern_match = self.match_pattern_plate(text_cleaned)
                    score = 0
                    if pattern_match:
                        score += 10  # приоритет полному совпадению с шаблоном
                    score += len(text_cleaned)  # приоритет длине

                    if score > best_score:
                        best_score = score
                        best_result = pattern_match or text_cleaned

        # Альтернатива — попытаться распознать из необработанного куска изображения через YOLO
        if not best_result:
            best_result = self.__recognize_text(image, detection_results)

        return best_result

    def __detection_plate(self, image) -> 'ultralytics.engine.results.Boxes':
        """
        Обнаруживает номерные знаки на изображении с помощью модели YOLO.

        :param image: Изображение
        :return: объект с координатами обнаруженных объектов на изображении
        """
        return self.__model_detection(image, device=self.device)

    def draw_bbox(self, bbox, frame, number):
        # Извлечение координат
        (top_left, top_right, bottom_right, bottom_left) = bbox
        top_left = tuple(map(int, top_left))
        bottom_right = tuple(map(int, bottom_right))

        # Отрисовка прямоугольника
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
        [x1, y1], [x2, y2], [x3, y3], [x4, y4] = bbox
        # Добавление текста
        label = f"{x1}, {y1}: Number: {number}"
        cv2.putText(frame, label, (top_left[0], top_left[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    def clean_string(self, input_string):
        allowed_chars = self.__alf_rus + self.__alf_eng + self.__nums
        pattern = f"[^{re.escape(allowed_chars)}]"
        return re.sub(pattern, '', input_string)

    def __recognize_text(self, frame, results_detection) -> str:
        """
        Распознает текст на выделенной модели YOLO области номерного знака.
        :param frame: Исходное изображение
        :param results_detection: результаты обнаружения номерных знаков YOLO
        :return: строка с распознанным номером или None, если номер не распознан
        """

        def replace_russian_with_english(plate: str) -> str:
            # Словарь для замены русских букв на английские
            translation_map = {
                'А': 'A', 'В': 'B', 'Е': 'E', 'К': 'K', 'М': 'M', 'Н': 'H',
                'О': 'O', 'Р': 'P', 'С': 'C', 'Т': 'T', 'У': 'Y', 'Х': 'X'
            }

            # Замена символов
            translated_plate = ''.join(translation_map.get(char, char) for char in plate)

            return translated_plate

        for box in results_detection[0].boxes:  # Результаты для первого изображения
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cropped_x1, cropped_y1, cropped_x2, cropped_y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(frame, (cropped_x1, cropped_y1), (cropped_x2, cropped_y2), (0, 255, 0), 2)
            cropped = frame.copy()
            cropped = cropped[cropped_y1:cropped_y2, cropped_x1:cropped_x2]

            # Применяем OCR
            results = self.__model_recognize.readtext(cropped)

            # Проверка каждого распознанного текста
            recog_number, recog_region, recog_plate = None, None, None
            txt = None
            print(results)
            for (bbox, text, prob) in results:

                text = self.clean_string(text.replace(' ', '').upper())
                txt = replace_russian_with_english(text[::])
                print(text)

                number = self.match_pattern_number(text)
                region = self.match_pattern_region(text)
                plate = self.match_pattern_plate(text)

                if not (number is None):
                    bbox_number = bbox
                    recog_number = number

                if not (region is None):
                    bbox_region = bbox
                    recog_region = region

                if not (region is None):
                    recog_plate = plate

            # number = replace_russian_with_english(label) + "  |  " + label_plate
            number = f'{replace_russian_with_english(str(recog_number))}'
            cv2.putText(frame, number, (cropped_x1 - 50, cropped_y1),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2)
