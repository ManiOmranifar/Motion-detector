import sys
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, 
                             QHBoxLayout, QWidget, QDialog, QPushButton)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False, 
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
)

# =====================================================================
# تابع کمکی محاسبه فاصله histogram رنگ (Bhattacharyya distance)
# =====================================================================
def color_histogram_distance(img1, img2, bins=32):
    # تبدیل به HSV برای رنگ‌سنجی بهتر
    hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    
    hist1 = cv2.calcHist([hsv1], [0,1], None, [bins, bins], [0,180, 0,256])
    hist2 = cv2.calcHist([hsv2], [0,1], None, [bins, bins], [0,180, 0,256])
    
    cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
    
    dist = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
    return dist

# =====================================================================
# ترد اسکنر محیط (مدیریت دوربین، تشخیص حرکت و بافر رم با Debounce پیشرفته)
# =====================================================================
class ScannerThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    motion_detected_signal = pyqtSignal(list, tuple)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self._active = True
        self.frame_buffer = deque(maxlen=20)  # بافر 20 فریم برای نمایش زنده و آنالیز
        self.cap = None
        self.backSub = None
        # for debounce motion confirmation
        self.debounce_frames = deque(maxlen=7)  # 7 فریم برای تایید حرکت
        self.debounce_bbox = None
        self.debounce_active = False
        self.debounce_color_ref = None

    def run(self):
        while self._run_flag:
            if not self._active:
                if self.cap is not None:
                    self.cap.release()
                    self.cap = None
                self.msleep(200)
                continue

            if self.cap is None:
                self.cap = cv2.VideoCapture(0)
                self.backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

            ret, frame = self.cap.read()
            if not ret:
                continue

            self.frame_buffer.append(frame.copy())
            fgMask = self.backSub.apply(frame)
            fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

            if len(self.frame_buffer) >= 20 and not self.debounce_active:
                contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    if cv2.contourArea(contour) > 50:
                        x, y, w, h = cv2.boundingRect(contour)

                        # شروع debounce بافر
                        self.debounce_active = True
                        self.debounce_bbox = (x, y, w, h)
                        self.debounce_frames.clear()
                        self.debounce_frames.append(frame.copy())
                        # رنگ مرجع برای تغییر رنگ ناحیه
                        self.debounce_color_ref = frame[y:y+h, x:x+w].copy()
                        break

            elif self.debounce_active:
                # جمع آوری فریم‌ها برای تایید تغییر رنگ
                self.debounce_frames.append(frame.copy())

                x, y, w, h = self.debounce_bbox

                # بررسی تغییر رنگ در ناحیه مورد نظر نسبت به فریم مرجع
                current_patch = frame[y:y+h, x:x+w]
                color_dist = color_histogram_distance(self.debounce_color_ref, current_patch)

                # اگر فاصله رنگ کمتر از 0.3 شد یعنی تغییر قابل توجه است (0 معتبر است، 1 خیلی متفاوت)
                # البته ممکنه نور و سایه و نویز وجود داشته باشه؛ بنابراین شرایط زیر را اضافه می‌کنیم:
                # باید رنگ تغییر ثابت یا در حال افزایش باشد در چند فریم اخیر.

                # ذخیره مقدار فاصله اخیر
                if not hasattr(self, 'color_dist_history'):
                    self.color_dist_history = deque(maxlen=5)
                self.color_dist_history.append(color_dist)

                # شرط تایید حرکت: فاصله رنگ باید بالاتر از 0.2 و رو به افزایش باشد (ثبات تغییر رنگ)
                dist_array = np.array(self.color_dist_history)
                increasing = all(dist_array[i] <= dist_array[i+1]+0.01 for i in range(len(dist_array)-1))
                color_change_detected = (color_dist > 0.5 and increasing)

                # همراه با تغییر رنگ، بررسی کانتورهای جدید نیز انجام شود:
                fg_roi = fgMask[y:y+h, x:x+w]
                contours_roi, _ = cv2.findContours(fg_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                motion_area = sum(cv2.contourArea(c) for c in contours_roi)
                motion_detected = motion_area > 50  # آستانه کوچک‌تر برای تایید

                # تصمیم نهایی برای ادامه debounce یا اعلام تشخیص:
                if color_change_detected or motion_detected:
                    # پس از 7 فریم تایید، ارسال به پنجره آنالیز
                    if len(self.debounce_frames) >= 7:
                        self._active = False
                        self.motion_detected_signal.emit(list(self.debounce_frames), self.debounce_bbox)
                        self.debounce_active = False
                        self.color_dist_history.clear()
                else:
                    # اگر تغییر رنگ یا حرکت تأیید نشد، debounce را لغو کن
                    if len(self.debounce_frames) >= 7:
                        # لغو debounce و برگشت به اسکن زنده
                        self.debounce_active = False
                        self.color_dist_history.clear()

            # ارسال فریم زنده برای نمایش
            self.change_pixmap_signal.emit(frame)

        if self.cap is not None:
            self.cap.release()

    def restart_project(self):
        self.frame_buffer.clear()
        self._active = True
        self.debounce_active = False
        self.color_dist_history = deque(maxlen=5)

    def stop(self):
        self._run_flag = False
        self.wait()


# =====================================================================
# پنجره آنالیز (پردازش دوطرفه، هوش مصنوعی مدی پایپ، لبه یابی و امتیازدهی)
# =====================================================================
class AnalysisWindow(QDialog):
    def __init__(self, frames, bbox):
        super().__init__()
        self.setWindowTitle("AI-POWERED BI-DIRECTIONAL TARGET ANALYSIS")
        self.setFixedSize(900, 600)
        self.setStyleSheet("background-color: #050505; color: #00FF00;")
        
        # ترکیب فریم‌ها برای اسکن دوطرفه (20 فریم مستقیم + 20 فریم معکوس = 40 مرحله)
        self.frames_forward = list(frames)
        self.frames_backward = list(reversed(frames))
        self.all_frames_to_process = self.frames_forward + self.frames_backward 
        
        self.original_bbox = bbox 
        self.idx = 0
        self.best_score = -1
        self.best_frame_data = None
        
        self.init_ui()
        
        # تایمر برای نمایش زنده مراحل آنالیز با سرعت بالا
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.step_analysis)
        self.timer.start(50) # هر 50 میلی ثانیه یک فریم پردازش می شود

    def init_ui(self):
        layout = QVBoxLayout()
        
        self.info_label = QLabel("INITIALIZING AI & BI-DIRECTIONAL SCAN...")
        self.info_label.setFont(QFont("Consolas", 14, QFont.Bold))
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet("padding: 10px; background-color: #111; border: 1px solid #333;")
        
        # نمایشگرها
        displays = QHBoxLayout()
        self.left_view = QLabel()
        self.right_view = QLabel()
        
        for v in [self.left_view, self.right_view]:
            v.setFixedSize(400, 350)
            v.setAlignment(Qt.AlignCenter)
            v.setStyleSheet("border: 2px solid #004400; background-color: #000;")
            
        displays.addWidget(self.left_view)
        displays.addWidget(self.right_view)
        
        # لیبل عناوین نمایشگرها
        titles = QHBoxLayout()
        lbl_left = QLabel("GLOBAL VIEW (FULL SCENE)")
        lbl_right = QLabel("LOCAL VIEW (AI ZOOM TARGET)")
        for lbl in [lbl_left, lbl_right]:
            lbl.setFont(QFont("Consolas", 12))
            lbl.setAlignment(Qt.AlignCenter)
        titles.addWidget(lbl_left)
        titles.addWidget(lbl_right)
        
        # دکمه ری‌استارت
        self.btn_restart = QPushButton("RE-SCAN ENVIRONMENT")
        self.btn_restart.setFont(QFont("Consolas", 16, QFont.Bold))
        self.btn_restart.setStyleSheet("""
            QPushButton { background-color: #330000; color: #FF3333; border: 2px solid #FF3333; padding: 15px; }
            QPushButton:hover { background-color: #FF3333; color: black; }
        """)
        self.btn_restart.hide()
        self.btn_restart.clicked.connect(self.accept) # بستن دیالوگ و بازگشت به 메ین
        
        layout.addWidget(self.info_label)
        layout.addLayout(titles)
        layout.addLayout(displays)
        layout.addWidget(self.btn_restart)
        self.setLayout(layout)

    def advanced_edge_detection(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        contrast = clahe.apply(gray)
        blurred = cv2.bilateralFilter(contrast, d=9, sigmaColor=75, sigmaSpace=75)
        edges = cv2.Canny(blurred, 50, 150)
        return edges

    def step_analysis(self):
        if self.idx < len(self.all_frames_to_process):
            frame = self.all_frames_to_process[self.idx]
            mode = "FORWARD (Old->New)" if self.idx < 20 else "BACKWARD (New->Old)"

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            human_detected = False
            human_bonus_score = 0

            x, y, w, h = self.original_bbox

            if results.pose_landmarks:
                human_detected = True
                human_bonus_score = 1000000

                landmarks = results.pose_landmarks.landmark
                h_f, w_f, _ = frame.shape

                all_x = [int(lm.x * w_f) for lm in landmarks]
                all_y = [int(lm.y * h_f) for lm in landmarks]

                pad = 40
                x1 = max(0, min(all_x) - pad)
                y1 = max(0, min(all_y) - pad)
                x2 = min(w_f, max(all_x) + pad)
                y2 = min(h_f, max(all_y) + pad)

                x, y, w, h = x1, y1, x2 - x1, y2 - y1
            else:
                pad = 30
                x = max(0, x - pad)
                y = max(0, y - pad)
                w = min(frame.shape[1] - x, w + 2 * pad)
                h = min(frame.shape[0] - y, h + 2 * pad)

            zoomed = frame[y:y + h, x:x + w].copy()
            if zoomed.size > 0:
                edges = self.advanced_edge_detection(zoomed)

                edge_density = cv2.countNonZero(edges)
                current_score = edge_density + (w * h) + human_bonus_score

                if current_score > self.best_score:
                    self.best_score = current_score
                    self.best_frame_data = {
                        'frame': frame.copy(),
                        'zoomed': zoomed.copy(),
                        'x': x, 'y': y, 'w': w, 'h': h,
                        'is_human': human_detected,
                        'mode': mode,
                        'frame_num': (self.idx % 20) + 1
                    }

                status_text = "HUMAN IDENTIFIED" if human_detected else "SCANNING BLOB..."
                self.info_label.setText(f"{mode} [{self.idx+1}/40] | AI STATUS: {status_text} | SCORE: {current_score}")

                edge_view = np.zeros_like(zoomed)
                edge_view[edges > 0] = [0, 255, 0]

                global_view = (frame.copy() * 0.3).astype(np.uint8)
                cv2.rectangle(global_view, (x, y), (x + w, y + h), (0, 255, 0), 2)
                global_view[y:y + h, x:x + w] = cv2.addWeighted(zoomed, 0.7, edge_view, 0.3, 0)

                self.render_frames(global_view, edge_view)

            self.idx += 1
        else:
            self.finalize_analysis()

    def finalize_analysis(self):
        self.timer.stop()
        if self.best_frame_data:
            d = self.best_frame_data
            final_frame = d['frame']
            final_zoom = d['zoomed']
            bx, by, bw, bh = d['x'], d['y'], d['w'], d['h']

            self._draw_pro_target(final_zoom, 0, 0, bw, bh)
            self._draw_pro_target(final_frame, bx, by, bw, bh)

            subject_type = "HUMAN" if d['is_human'] else "UNKNOWN OBJECT"
            self.info_label.setText(
                f"TARGET SECURED: {subject_type} | BEST RESULT FOUND IN {d['mode']} (FRAME {d['frame_num']})")
            self.info_label.setStyleSheet("padding: 10px; background-color: #004400; color: #FFF; font-weight: bold;")

            self.render_frames(final_frame, final_zoom)
        else:
            self.info_label.setText("ANALYSIS FAILED: NO VALID TARGET RECOGNIZED.")
            self.info_label.setStyleSheet("padding: 10px; background-color: #550000; color: #FFF;")

        self.btn_restart.show()

    def _draw_pro_target(self, img, x, y, w, h):
        color = (0, 255, 0)
        thick = 2
        l = min(40, int(w * 0.2))

        cv2.line(img, (x, y), (x + l, y), color, thick)
        cv2.line(img, (x, y), (x, y + l), color, thick)
        cv2.line(img, (x + w, y), (x + w - l, y), color, thick)
        cv2.line(img, (x + w, y), (x + w, y + l), color, thick)
        cv2.line(img, (x, y + h), (x + l, y + h), color, thick)
        cv2.line(img, (x, y + h), (x, y + h - l), color, thick)
        cv2.line(img, (x + w, y + h), (x + w - l, y + h), color, thick)
        cv2.line(img, (x + w, y + h), (x + w, y + h - l), color, thick)

        cx, cy = x + w // 2, y + h // 2
        cv2.drawMarker(img, (cx, cy), color, cv2.MARKER_CROSS, 20, 1)
        cv2.circle(img, (cx, cy), 30, color, 1)

    def render_frames(self, left_img, right_img):
        for img, label in [(left_img, self.left_view), (right_img, self.right_view)]:
            if img is None or img.size == 0: continue
            resized = cv2.resize(img, (400, 350))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            h, w, c = rgb.shape
            qimg = QImage(rgb.data, w, h, w * c, QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(qimg))


# =====================================================================
# اپلیکیشن اصلی (رابط کاربری مادر)
# =====================================================================
class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SMART VISION AI - STANDBY MODE")
        self.setFixedSize(800, 600)
        self.setStyleSheet("background-color: black;")

        self.display = QLabel()
        self.display.setAlignment(Qt.AlignCenter)
        self.setCentralWidget(self.display)

        self.scanner = ScannerThread()
        self.scanner.change_pixmap_signal.connect(self.update_screen)
        self.scanner.motion_detected_signal.connect(self.on_detection)
        self.scanner.start()

    def update_screen(self, img):
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        qimg = QImage(rgb_img.data, w, h, ch * w, QImage.Format_RGB888)
        self.display.setPixmap(QPixmap.fromImage(qimg).scaled(self.display.size(), Qt.KeepAspectRatio))

    def on_detection(self, frames, bbox):
        self.hide()
        analysis_dialog = AnalysisWindow(frames, bbox)
        if analysis_dialog.exec_():
            self.show()
            self.setFixedSize(800, 600)
            self.scanner.restart_project()

    def closeEvent(self, event):
        self.scanner.stop()
        event.accept()


# =====================================================================
# اجرای برنامه
# =====================================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())
