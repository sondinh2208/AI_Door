"""
╔══════════════════════════════════════════════════════════════════════╗
║  ai_server.py – Trạm AI Nhận diện Khuôn mặt Trung tâm                ║
║  Mô tả: Giao tiếp MQTT, luồng xử lý nhận diện khuôn mặt (DeepFace),  ║
║         tích hợp chống giả mạo (FASNet), CLAHE cho điều kiện chói lóa.║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os
import cv2
import base64
import time
import ssl
import queue
import threading
import numpy as np
import torch
import paho.mqtt.client as mqtt
from deepface import DeepFace
from dotenv import load_dotenv

# ====================================================================
# 1. THIẾT LẬP MÔI TRƯỜNG AI TRƯỚC KHI KHỞI TẠO
# ====================================================================
os.environ["KERAS_BACKEND"] = "torch"
os.environ["YOLO_MIN_DETECTION_CONFIDENCE"] = "0.15"

load_dotenv()

# ====================================================================
# 2. CẤU HÌNH HỆ THỐNG
# ====================================================================
class Config:
    # --- MQTT ---
    MQTT_BROKER = os.getenv("MQTT_BROKER")
    MQTT_PORT = int(os.getenv("MQTT_PORT", 8883))
    MQTT_USERNAME = os.getenv("MQTT_USERNAME")
    MQTT_PASSWORD = os.getenv("MQTT_PASSWORD")
    TOPIC_CAMERA = "haui/smartdoor/camera"
    TOPIC_CONTROL = "haui/smartdoor/control"

    # --- AI & Nhận Diện ---
    MODEL_NAME = "Facenet512"
    DISTANCE_METRIC = "cosine"
    DETECTOR_BACKEND = "yolov8n"
    FACES_DB_PATH = "faces_db/"
    ANTI_SPOOFING = True
    DISTANCE_THRESHOLD = 0.5  # Ngưỡng tối ưu nhận diện (0.38 - 0.5)

    # --- Xử lý Ảnh (Anti-Glare & Clarity) ---
    ENABLE_CLAHE = True
    CLAHE_CLIP_LIMIT = 2.0
    CLAHE_GRID_SIZE = (8, 8)

    # --- Sleep/Wake Config ---
    AI_SCAN_INTERVAL = 4.0  # Tần suất AI nhận diện (4 giây/lần)
    SLEEP_TIMEOUT = 15.0    # 15 giây không có ai sẽ gửi lệnh ngủ

    # --- Camera Transform ---
    ROTATE_CAMERA = 0
    FLIP_HORIZONTAL = False
    FLIP_VERTICAL = False


# ====================================================================
# 3. TRẠNG THÁI TOÀN CỤC
# ====================================================================
frame_queue = queue.Queue(maxsize=1)
last_ai_scan = 0
system_awake = False
last_face_time = 0
last_sleep_time = 0


# ====================================================================
# 4. HÀM HỖ TRỢ XỬ LÝ ẢNH
# ====================================================================
def decode_base64_to_image(b64_string: str) -> np.ndarray:
    """Giải mã chuỗi Base64 thành ảnh OpenCV (NumPy BGR array)."""
    jpeg_bytes = base64.b64decode(b64_string)
    np_array = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    return cv2.imdecode(np_array, cv2.IMREAD_COLOR)

def apply_camera_transformations(image: np.ndarray) -> np.ndarray:
    """Điều chỉnh chiều xoay và lật khung hình của camera."""
    if image is None or image.size == 0:
        return image

    if Config.ROTATE_CAMERA == 180:
        image = cv2.rotate(image, cv2.ROTATE_180)
    elif Config.ROTATE_CAMERA == 90:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif Config.ROTATE_CAMERA == 270:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    if Config.FLIP_HORIZONTAL:
        image = cv2.flip(image, 1)
    if Config.FLIP_VERTICAL:
        image = cv2.flip(image, 0)

    return image

def apply_clahe(image_bgr: np.ndarray) -> np.ndarray:
    """Giảm lóa sáng bằng phương pháp CLAHE trên không gian LAB."""
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=Config.CLAHE_CLIP_LIMIT, tileGridSize=Config.CLAHE_GRID_SIZE)
    l_clahe = clahe.apply(l_channel)
    
    enhanced_lab = cv2.merge((l_clahe, a_channel, b_channel))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

def enhance_image_clarity(image_bgr: np.ndarray) -> np.ndarray:
    """Tăng cường độ nét ảnh (Unsharp Masking)."""
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr
    gaussian = cv2.GaussianBlur(image_bgr, (0, 0), sigmaX=1.5)
    return cv2.addWeighted(image_bgr, 1.35, gaussian, -0.35, 0)


# ====================================================================
# 5. HÀM AI & NHẬN DIỆN KHUÔN MẶT
# ====================================================================
def warmup_model():
    """Khởi động GPU/CPU và nạp sẵn model DeepFace."""
    if torch.cuda.is_available():
        print(f"[GPU] ✅ CUDA sẵn sàng: {torch.cuda.get_device_name(0)}")
    else:
        print("[GPU] ⚠️ CUDA KHÔNG khả dụng! Đang chạy trên CPU.")

    print("[AI] 🧠 Đang nạp mô hình...")
    dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)

    try:
        DeepFace.represent(img_path=dummy_img, model_name=Config.MODEL_NAME, detector_backend=Config.DETECTOR_BACKEND, enforce_detection=False)
        print("[AI] ✅ Facenet512 hoàn tất.")
    except Exception as e:
        print(f"[AI] ⚠️ Lỗi nạp Facenet512: {e}")

    if Config.ANTI_SPOOFING:
        try:
            DeepFace.extract_faces(img_path=dummy_img, detector_backend=Config.DETECTOR_BACKEND, anti_spoofing=True, enforce_detection=False)
            print("[AI] ✅ Anti-Spoofing (FASNet) hoàn tất.")
        except Exception as e:
            print(f"[AI] ⚠️ Lỗi Anti-Spoofing: {e}")

def check_anti_spoofing(image: np.ndarray) -> tuple:
    """Kiểm tra ảnh là người thật hay giả mạo."""
    faces = DeepFace.extract_faces(
        img_path=image, detector_backend=Config.DETECTOR_BACKEND, 
        anti_spoofing=True, enforce_detection=True
    )
    if faces and len(faces) > 0:
        return faces[0].get("is_real", False), faces[0].get("antispoof_score", 0.0)
    return False, 0.0

def publish_command(client: mqtt.Client, command: str, message: str, elapsed: float, score: float = None, distance: float = None):
    """Gửi lệnh MQTT và in log chi tiết."""
    print(message)
    if score is not None:
        print(f"     🛡️  Anti-Spoof : {'REAL' if score >= 0.5 else 'FAKE'} (score: {score:.4f})")
    if distance is not None:
        print(f"     📐 Khoảng cách: {distance:.6f} (Ngưỡng: {Config.DISTANCE_THRESHOLD})")
    print(f"     ⏱️  Xử lý     : {elapsed:.4f}s")
    print(f"     🔒 Lệnh MQTT  : {command}")
    print("-" * 55)
    client.publish(Config.TOPIC_CONTROL, command)

def process_face_recognition(image: np.ndarray, client: mqtt.Client):
    """Thực thi pipeline nhận diện toàn diện."""
    global last_face_time
    start_time = time.time()
    face_img = image

    try:
        # Bước 1: Anti-Spoofing
        if Config.ANTI_SPOOFING:
            try:
                is_real, spoof_score = check_anti_spoofing(image)
            except ValueError:
                # Nếu ảnh lóa sáng, thử lại với CLAHE
                if Config.ENABLE_CLAHE:
                    clahe_img = apply_clahe(image)
                    is_real, spoof_score = check_anti_spoofing(clahe_img)
                    face_img = clahe_img
                else:
                    raise

            if not is_real:
                publish_command(client, "DENIED", "[AI] ⚠️ TỪ CHỐI: Phát hiện giả mạo!", time.time() - start_time, spoof_score)
                return

        # Bước 2: So khớp đặc trưng
        enhanced_face_img = enhance_image_clarity(face_img)
        results = DeepFace.find(
            img_path=enhanced_face_img, db_path=Config.FACES_DB_PATH,
            model_name=Config.MODEL_NAME, detector_backend=Config.DETECTOR_BACKEND,
            distance_metric=Config.DISTANCE_METRIC, enforce_detection=False, silent=True
        )

        elapsed = time.time() - start_time
        if results and len(results) > 0 and not results[0].empty:
            best_match = results[0].iloc[0]
            dist_col = f"{Config.MODEL_NAME}_{Config.DISTANCE_METRIC}"
            distance = float(best_match.get(dist_col, best_match.get("distance", 1.0)))
            
            # Trích xuất tên người dùng
            identity_path = str(best_match["identity"])
            user_name = os.path.basename(os.path.dirname(identity_path))
            if not user_name or user_name == os.path.basename(Config.FACES_DB_PATH.rstrip("/\\")):
                user_name = os.path.splitext(os.path.basename(identity_path))[0]

            if distance <= Config.DISTANCE_THRESHOLD:
                msg = f"[AI] ✅ NHẬN DIỆN THÀNH CÔNG: {user_name}"
                publish_command(client, "OPEN_FACE", msg, elapsed, spoof_score if Config.ANTI_SPOOFING else 1.0, distance)
            else:
                msg = f"[AI] ❌ KẺ LẠ MẶT (Gần khớp với {user_name} nhưng vượt ngưỡng)"
                publish_command(client, "DENIED", msg, elapsed, spoof_score if Config.ANTI_SPOOFING else 1.0, distance)
        else:
            publish_command(client, "DENIED", "[AI] ❌ KẺ LẠ MẶT - Không có dữ liệu khớp", elapsed)

        # Cập nhật thời gian thấy mặt để không bị Sleep
        last_face_time = time.time()

    except ValueError:
        print("[AI] ⏳ Đang chờ người dùng đứng vào camera...")
        print("-" * 55)
        client.publish(Config.TOPIC_CONTROL, "NO_FACE")
    except Exception as e:
        print(f"[AI] ⚠️ Lỗi pipeline xử lý: {e}")
        print("-" * 55)


# ====================================================================
# 6. GIAO DIỆN & LUỒNG XỬ LÝ SỰ KIỆN CAMERA
# ====================================================================
def handle_camera_hotkeys(key, image):
    """Xử lý thao tác phím trên cửa sổ Live Camera."""
    if key in (ord('r'), ord('R')):
        rotations = {180: 0, 0: 90, 90: 270, 270: 180}
        Config.ROTATE_CAMERA = rotations.get(Config.ROTATE_CAMERA, 180)
        print(f"[CAM] 🔄 Đổi góc xoay -> {Config.ROTATE_CAMERA}°")
    elif key in (ord('f'), ord('F')):
        Config.FLIP_HORIZONTAL = not Config.FLIP_HORIZONTAL
        print(f"[CAM] 🔄 Lật ngang -> {'BẬT' if Config.FLIP_HORIZONTAL else 'TẮT'}")
    elif key in (ord('v'), ord('V')):
        Config.FLIP_VERTICAL = not Config.FLIP_VERTICAL
        print(f"[CAM] 🔄 Lật dọc -> {'BẬT' if Config.FLIP_VERTICAL else 'TẮT'}")
    elif key in (ord('s'), ord('S')):
        target_user = "Dinh Cong Son"
        user_folder = os.path.join(Config.FACES_DB_PATH, target_user)
        os.makedirs(user_folder, exist_ok=True)

        filepath = os.path.join(user_folder, f"sample_{int(time.time())}.jpg")
        cv2.imwrite(filepath, image)

        # Xóa cache .pkl để tự động nạp lại
        for f in os.listdir(Config.FACES_DB_PATH):
            if f.endswith(".pkl"):
                os.remove(os.path.join(Config.FACES_DB_PATH, f))
        print(f"[DB] 📸 Đã lưu ảnh mẫu vào: {filepath}")

def process_frames_worker(client: mqtt.Client):
    """Luồng worker nền để quét AI độc lập không block MQTT."""
    global last_ai_scan, system_awake, last_face_time, last_sleep_time
    print("[THREAD] 🔄 Luồng xử lý AI đã sẵn sàng. Đang chờ Wake Up...")

    while True:
        if not system_awake:
            time.sleep(0.1)
            continue
            
        if time.time() - last_face_time > Config.SLEEP_TIMEOUT:
            print(f"\n[SYS] 🌙 Hết thời gian {Config.SLEEP_TIMEOUT}s. Gửi lệnh SLEEP!")
            client.publish(Config.TOPIC_CONTROL, "SLEEP")
            system_awake = False
            last_sleep_time = time.time()
            cv2.destroyAllWindows()
            while not frame_queue.empty():
                frame_queue.get()
            continue

        try:
            raw_image = frame_queue.get(timeout=0.5)
            image = apply_camera_transformations(raw_image)
            cv2.imshow("ESP32-S3 Live Camera (Awake)", image)

            handle_camera_hotkeys(cv2.waitKey(1) & 0xFF, image)

            if time.time() - last_ai_scan >= Config.AI_SCAN_INTERVAL:
                print(f"[AI] 🖼️  Kích thước ảnh: {image.shape[1]}x{image.shape[0]}")
                process_face_recognition(image, client)
                last_ai_scan = time.time()

        except queue.Empty:
            cv2.waitKey(1)
        except Exception as e:
            print(f"[THREAD] ❌ Lỗi luồng: {e}")


# ====================================================================
# 7. MQTT CLIENT
# ====================================================================
def on_mqtt_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print("[MQTT] ✅ Kết nối thành công!")
        client.subscribe(Config.TOPIC_CAMERA)
        client.subscribe(Config.TOPIC_CONTROL)
        print("[AI] 🔍 Hệ thống sẵn sàng nhận diện...\n" + "=" * 55)
        client.publish(Config.TOPIC_CONTROL, "READY")
    else:
        print(f"[MQTT] ❌ Kết nối thất bại, rc: {rc}")

def on_mqtt_message(client, userdata, msg):
    global system_awake, last_face_time, last_sleep_time
    try:
        topic = msg.topic
        if topic == Config.TOPIC_CONTROL:
            command = msg.payload.decode('utf-8')
            if command == "WAKE_UP":
                print("\n[MQTT] 🔔 AI SERVER ĐÃ THỨC DẬY (Nhận WAKE_UP từ LM393)!")
                system_awake = True
                last_face_time = time.time()
                
        elif topic == Config.TOPIC_CAMERA:
            if not system_awake:
                # Tránh tình trạng lặp: Bỏ qua những frame ảnh cũ còn sót lại trong mạng ngay sau khi vừa gửi lệnh SLEEP
                if time.time() - last_sleep_time < 2.0:
                    return

                print("\n[MQTT] ⚠️ Bị lỡ mất lệnh WAKE_UP! Tự động THỨC DẬY vì có ảnh đang gửi tới!")
                system_awake = True
                last_face_time = time.time()
                
            # Giải mã ảnh
            image = decode_base64_to_image(msg.payload.decode('utf-8'))
            if image is not None and image.size > 0:
                try:
                    frame_queue.put_nowait(image)
                except queue.Full:
                    try: frame_queue.get_nowait()
                    except queue.Empty: pass
                    frame_queue.put_nowait(image)
    except Exception as e:
        pass

def create_mqtt_client() -> mqtt.Client:
    client = mqtt.Client(client_id="AI_Server_RTX4050", protocol=mqtt.MQTTv5)
    client.username_pw_set(Config.MQTT_USERNAME, Config.MQTT_PASSWORD)
    client.tls_set(cert_reqs=ssl.CERT_REQUIRED, tls_version=ssl.PROTOCOL_TLSv1_2)
    client.on_connect = on_mqtt_connect
    client.on_message = on_mqtt_message
    return client


# ====================================================================
# 8. HÀM MAIN
# ====================================================================
if __name__ == "__main__":
    print("=" * 55)
    print("  AI SERVER – Hệ thống Khóa cửa Thông minh")
    print("=" * 55)

    os.makedirs(Config.FACES_DB_PATH, exist_ok=True)
    warmup_model()

    mqtt_client = create_mqtt_client()
    mqtt_client.connect(Config.MQTT_BROKER, Config.MQTT_PORT)
    
    threading.Thread(target=process_frames_worker, args=(mqtt_client,), daemon=True).start()

    try:
        mqtt_client.loop_forever()
    except KeyboardInterrupt:
        print("\n[SYS] 🛑 Nhận tín hiệu ngắt (Ctrl+C).")
    finally:
        mqtt_client.disconnect()
        cv2.destroyAllWindows()
