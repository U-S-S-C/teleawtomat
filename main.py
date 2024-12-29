import cv2
import numpy as np
from videocapture import ScreenCapture  # Импортируем класс для захвата экрана
from frame_display import update_frame  # Импортируем функцию для отрисовки
from scanner import Scanner  # Импортируем сканер

def on_trackbar(val):
    pass

def draw_settings_descriptions(frame):
    # Добавляем текстовые описания для каждого ползунка
    cv2.putText(frame, "WinSize: Size of the search window", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, "MaxLvl: Max pyramid levels for LK", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, "CritCount: Max iterations for criteria", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, "CritEps: Epsilon for criteria", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, "AutoSize: Automatic window size", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    # Этот параметр определяет размер окна, в котором будет производиться поиск соответствий для отслеживания точки. Больший размер окна может улучшить устойчивость к шуму, но также может замедлить процесс и сделать его менее точным для мелких деталей
    # Этот параметр задаёт количество уровней в пирамиде изображений, которые используются в алгоритме Lucas-Kanade для отслеживания. Большее количество уровней может улучшить отслеживание на изображениях с разными масштабами, но также увеличивает вычислительную нагрузку
    # Это максимальное количество итераций, которые алгоритм будет выполнять для достижения заданного критерия сходимости. Большее количество итераций может улучшить точность, но также увеличивает время выполнения
    # Этот параметр определяет порог точности, при достижении которого алгоритм считает, что решение найдено. Меньшее значение эпсилона может привести к более точным результатам, но также может увеличить количество итераций, необходимых для достижения сходимости
if __name__ == "__main__":
    # Создаем объект захвата экрана и сканера
    screen_capture = ScreenCapture(top=100, left=0, width=600, height=400)
    scanner = Scanner()
    lock_active = False  # Индикатор захвата точки

    # Создаем окно для настройки параметров
    cv2.namedWindow("Settings", cv2.WINDOW_NORMAL)  # Позволяет изменять размер окна
    cv2.resizeWindow("Settings", 800, 200)  # Увеличиваем высоту окна для текста

    cv2.createTrackbar("WinSize", "Settings", 15, 50, on_trackbar)
    cv2.createTrackbar("MaxLvl", "Settings", 2, 5, on_trackbar)  # Сокращенное название
    cv2.createTrackbar("CritCount", "Settings", 10, 50, on_trackbar)  # Сокращенное название
    cv2.createTrackbar("CritEps", "Settings", 3, 10, on_trackbar)  # Сокращенное название
    cv2.createTrackbar("AutoSize", "Settings", 0, 1, on_trackbar)  # Переключатель для автоматического размера

    while True:
        # Захват области экрана
        frame = screen_capture.capture_area()

        # Обновляем кадр в сканере
        scanner.update_frame(frame)

        # Обновляем параметры сканера
        win_size = cv2.getTrackbarPos("WinSize", "Settings")
        max_level = cv2.getTrackbarPos("MaxLvl", "Settings")
        criteria_count = cv2.getTrackbarPos("CritCount", "Settings")
        criteria_eps = cv2.getTrackbarPos("CritEps", "Settings") / 100.0
        auto_size = cv2.getTrackbarPos("AutoSize", "Settings")

        scanner.lk_params['maxLevel'] = max_level
        scanner.lk_params['criteria'] = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, criteria_count, criteria_eps)

        # Если точка захвачена, отслеживаем её
        if lock_active:
            result = scanner.track()
            if result is not None:  # Проверяем, вернул ли track что-то
                tracked_point, contour_size = result
                x, y = tracked_point
                if auto_size and contour_size is not None:
                    search_size = min(max(4, contour_size), 100)
                else:
                    search_size = win_size
                half_size = search_size // 2
                cv2.rectangle(frame, (int(x) - half_size, int(y) - half_size), (int(x) + half_size, int(y) + half_size), (0, 255, 0), 2)  # Зелёная рамка
            else:
                lock_active = False  # Сбрасываем индикатор захвата
                print("Lost tracking of the point.")

        # Обновляем кадр на экране
        update_frame(frame)

        # Рисуем описания ползунков
        settings_frame = np.zeros((200, 800, 3), dtype=np.uint8)
        draw_settings_descriptions(settings_frame)
        cv2.imshow("Settings", settings_frame)

        # Обработка клавиш
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Выход при нажатии 'q'
            break
        elif key == ord('l'):  # Захват самой тёмной точки
            if not lock_active:  # Если точка не захвачена
                result = scanner.lock_darkest_point()
                if result is not None:
                    locked_point, contour_size = result
                    lock_active = True
                    print(f"Locked on darkest point at {locked_point}.")
                else:
                    print("No suitable dark point found.")
            else:
                print("Point is already locked. Tracking active.")

    cv2.destroyAllWindows()
