# frame_display.py

import cv2

def update_frame(frame, window_name="Screen Capture"):
    """
    Отображает кадр в указанном окне.
    :param frame: Кадр (numpy array), который нужно отобразить.
    :param window_name: Название окна для отображения.
    """
    # Преобразуем кадр в RGB формат для корректного отображения
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    # Отображаем кадр
    cv2.imshow(window_name, frame)
