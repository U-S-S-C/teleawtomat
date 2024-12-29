import cv2
import numpy as np

class Scanner:
    def __init__(self):
        self.current_frame = None
        self.locked_point = None
        self.prev_gray_frame = None
        self.lk_params = dict(winSize=(15, 15), maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    def update_frame(self, frame):
        self.current_frame = frame

    def lock_darkest_point(self):
        if self.current_frame is None:
            return None

        gray_frame = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)

        # Применяем порог для выделения тёмных областей
        _, threshold = cv2.threshold(gray_frame, 100, 255, cv2.THRESH_BINARY_INV)

        # Находим контуры тёмных областей
        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Находим контур с максимальной площадью
        largest_contour = max(contours, key=cv2.contourArea)

        # Вычисляем центр масс этого контура
        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            return None

        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        # Вычисляем размер контура
        x, y, w, h = cv2.boundingRect(largest_contour)
        contour_size = max(w, h)

        self.locked_point = np.array([[cX, cY]], dtype=np.float32)
        self.prev_gray_frame = gray_frame

        return (cX, cY), contour_size

    def track(self):
        if self.current_frame is None or self.locked_point is None:
            return None

        gray_frame = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)

        # Используем оптический поток для отслеживания точки
        new_point, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray_frame, gray_frame, self.locked_point, None, **self.lk_params)

        if new_point is not None and status is not None and len(new_point) > 0 and status[0][0] == 1:
            self.locked_point = new_point
            self.prev_gray_frame = gray_frame

            # Находим контуры в текущем кадре
            _, threshold = cv2.threshold(gray_frame, 100, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                contour_size = max(w, h)

                # Определяем размер окна поиска
                if contour_size > 100:
                    search_size = 100
                else:
                    search_size = max(4, contour_size)

                # Обновляем размер окна поиска
                self.lk_params['winSize'] = (search_size, search_size)
            else:
                contour_size = None

            return (int(new_point[0][0]), int(new_point[0][1])), contour_size
        else:
            self.locked_point = None
            return None
