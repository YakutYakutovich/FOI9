import cv2
import numpy as np


# Упорядочивание точек
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


# Выравнивание изображения по четырём точкам
def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


# Извлечение объединённого квадрата областей из результатов EasyOCR
def extract_combined_quad(results):
    if not results:
        return None

    all_points = np.concatenate([np.array(r[0]) for r in results], axis=0).astype(np.float32)

    # Находим минимальный вращённый прямоугольник
    rotated_rect = cv2.minAreaRect(all_points)
    box = cv2.boxPoints(rotated_rect)
    return np.array(box, dtype="float32")


# Возврат вышедшего контура текста в пределы изображения
def clip_quad_to_image(quad, image_shape):
    h, w = image_shape[:2]
    quad_clipped = []

    for x, y in quad:
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        quad_clipped.append([x, y])

    return np.array(quad_clipped, dtype="float32")


# Поиск и выравнивание области с текстом на изображении
def straighten_with_easyocr(reader, image, contours=False, cnt=0):
    # Преобразуем в RGB, т.к. EasyOCR ожидает RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = reader.readtext(rgb_image, detail=1)

    if not results:
        return image

    if contours:
        # Создаём копию для отрисовки
        debug_image = image.copy()

        # Нарисуем прямоугольники всех найденных текстов
        for box, text, conf in results:
            pts = np.array(box, dtype=np.int32)
            cv2.polylines(debug_image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            # (опционально) рисуем текст
            x, y = pts[0]
            cv2.putText(debug_image, text, (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Сохраняем изображение с наложенными прямоугольниками
        cv2.imwrite(f"data/contours/{cnt}.jpg", debug_image)

    # Объединяем прямоугольники, получаем quad
    quad = extract_combined_quad(results)
    if quad is None:
        return image

    quad = clip_quad_to_image(quad, image.shape)

    if contours:
        debug_quad = image.copy()
        cv2.polylines(debug_quad, [quad.astype(int)], isClosed=True, color=(0, 0, 255), thickness=2)
        cv2.imwrite(f"data/quad_debug/{cnt}.jpg", debug_quad)

    warped = four_point_transform(image, quad)
    return warped


# Полная обработка изображения после YOLO до распознавания текста
def preprocessing_image(reader, img, contours=False, cnt=0):
    scale_factor = 4
    img = cv2.resize(img, (img.shape[1] * scale_factor, img.shape[0] * scale_factor), interpolation=cv2.INTER_CUBIC)

    # images = straighten_plate_from_image(img)
    img = straighten_with_easyocr(reader, img, contours, cnt)

    # # Дебаг после выравнивания
    # cv2.imwrite(f"data/debug/{cnt}.jpg", img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Увеличим изображение
    upscaled = cv2.resize(gray, (520, 112), interpolation=cv2.INTER_CUBIC)

    # Применим фильтр для повышения резкости
    sharpened = cv2.filter2D(upscaled, -1, kernel=np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))

    # Усилим контраст
    equalized = cv2.equalizeHist(sharpened)

    thresholded = cv2.threshold(equalized, 110, 255, cv2.THRESH_BINARY)[1]

    # Разрастание белых пикселей
    # inverted = cv2.bitwise_not(thresholded)
    kernel = np.ones((2, 2), np.uint8)
    dilated_white = cv2.dilate(thresholded, kernel, iterations=1)
    # final_result = cv2.bitwise_not(dilated_black)

    # # Дебаг перед очисткой по площади
    # cv2.imwrite(f"data/debug1/{cnt}.jpg", dilated_white)

    # Инвертируем цвет
    inverted = cv2.bitwise_not(dilated_white)

    # Находим связные компоненты
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inverted, connectivity=8)

    # Минимальная площадь скопления (можно настроить)
    min_area = 200
    # Квадрат для удаления
    min_width = 10
    min_height = 112

    # Создаём пустое изображение для результата
    cleaned = np.zeros_like(inverted)

    # Проходим по всем компонентам
    for i in range(1, num_labels):  # Пропускаем фон (i=0)
        area = stats[i, cv2.CC_STAT_AREA]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        if min_area <= area:
            if not (w <= min_width and h <= min_height):
                cleaned[labels == i] = 255

    cleaned = cv2.bitwise_not(cleaned)

    final_result = cleaned

    return final_result
