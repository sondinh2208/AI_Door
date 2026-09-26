"""
╔══════════════════════════════════════════════════════════════════════╗
║  ai_server.py – Trạm AI Nhận diện Khuôn mặt Trung tâm                ║
║  Tác giả : Kỹ sư Hệ thống Nhúng & AI                                 ║
║  Mô tả  : Nhận ảnh qua MQTT, luồng Threading + Queue non-blocking,  ║
║            Giao diện console dạng bảng chi tiết, trực quan, chuyên   ║
║            nghiệp với 4 trạng thái nhận diện rõ ràng.                ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os

# ====================================================================
# CẤU HÌNH BACKEND KERAS -> PYTORCH (PHẢI ĐẶT TRƯỚC KHI IMPORT DEEPFACE)
# ====================================================================
os.environ["KERAS_BACKEND"] = "torch"
# Hạ ngưỡng phát hiện của YOLO để nhận diện nhạy hơn trong điều kiện chói lóa/bóng đổ
os.environ["YOLO_MIN_DETECTION_CONFIDENCE"] = "0.15"

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

# Load cấu hình từ file .env
load_dotenv()

# ====================================================================
# HẰNG SỐ CẤU HÌNH
# ====================================================================
MQTT_BROKER   = os.getenv("MQTT_BROKER")
MQTT_PORT     = int(os.getenv("MQTT_PORT", 8883))
MQTT_USERNAME = os.getenv("MQTT_USERNAME")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD")

# Topic MQTT
TOPIC_CAMERA  = "haui/smartdoor/camera"        # Nhận ảnh từ camera
TOPIC_CONTROL = "haui/smartdoor/control"       # Gửi lệnh điều khiển

# Cấu hình AI
MODEL_NAME       = "Facenet512"    # Mô hình nhận diện (512-d embedding)
DISTANCE_METRIC  = "cosine"        # Metric đo khoảng cách vector đặc trưng
DETECTOR_BACKEND = "yolov8n"       # YOLOv8 Nano cực nhanh
FACES_DB_PATH    = "faces_db/"     # Thư mục cơ sở dữ liệu khuôn mặt
ANTI_SPOOFING    = True            # Bật kiểm tra chống giả mạo (FASNet)

# Ngưỡng khoảng cách cho Facenet512 + Cosine (0.38 tối ưu cho cả cự ly gần và cự ly xa 1 - 1.5m)
DISTANCE_THRESHOLD = 0.4
LAST_AI_SCAN = 0

# Cấu hình tiền xử lý chống lóa sáng (Cơ chế CLAHE Fallback trên không gian màu LAB)
ENABLE_CLAHE       = True          # Bật cơ chế cứu hộ CLAHE khi bị lóa sáng nặng
CLAHE_CLIP_LIMIT   = 2.0           # Giới hạn tương phản (2.0 - 3.0)
CLAHE_GRID_SIZE    = (8, 8)        # Kích thước lưới chia vùng cục bộ (8x8)

# Cấu hình chiều camera (sửa lỗi camera bị lắp ngược đầu)
ROTATE_CAMERA      = 0           # 0: Không xoay | 180: Xoay ngược 180° | 90: Xoay 90° | 270: Xoay 270°
FLIP_HORIZONTAL    = False         # True: Lật gương ngang (Trái <-> Phải)
FLIP_VERTICAL      = False         # True: Lật ngược dọc (Trên <-> Dưới)

# ====================================================================
# HÀNG ĐỢI KHUNG HÌNH (Frame Queue)
# ====================================================================
# maxsize=1: Chỉ lưu duy nhất 1 frame mới nhất, loại bỏ ảnh cũ khi bị trễ
frame_queue = queue.Queue(maxsize=1)


def warmup_model():
    """
    Khởi động nóng mô hình vào GPU RTX 4050 để các lần nhận diện sau đạt tốc độ tức thì.
    """
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[GPU] ✅ CUDA sẵn sàng: {gpu_name} ({vram_gb:.1f} GB VRAM)")
    else:
        print("[GPU] ⚠️  CUDA KHÔNG khả dụng! Đang chạy trên CPU.")

    print("[AI] 🧠 Đang nạp mô hình Facenet512 & Anti-Spoofing...")
    dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)

    try:
        DeepFace.represent(
            img_path=dummy_img,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False
        )
        print("[AI] ✅ Khởi tạo Facenet512 hoàn tất.")
    except Exception as e:
        print(f"[AI] ⚠️  Khởi tạo Facenet512: {e}")

    if ANTI_SPOOFING:
        try:
            DeepFace.extract_faces(
                img_path=dummy_img,
                detector_backend=DETECTOR_BACKEND,
                anti_spoofing=True,
                enforce_detection=False
            )
            print("[AI] ✅ Khởi tạo Anti-Spoofing (FASNet) hoàn tất.")
        except Exception as e:
            print(f"[AI] ⚠️  Khởi tạo Anti-Spoofing: {e}")


def decode_base64_to_image(b64_string: str) -> np.ndarray:
    """
    Giải mã chuỗi Base64 thành ảnh OpenCV (NumPy BGR array).
    """
    jpeg_bytes = base64.b64decode(b64_string)
    np_array = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    return image


def orient_image(image: np.ndarray) -> np.ndarray:
    """
    Điều chỉnh chiều khung hình (xoay 180°, 90°, lật gương ngang/dọc).
    Khắc phục trường hợp camera ESP32 bị lắp ngược đầu.
    """
    if image is None or image.size == 0:
        return image

    # Xoay khung hình
    if ROTATE_CAMERA == 180:
        image = cv2.rotate(image, cv2.ROTATE_180)
    elif ROTATE_CAMERA == 90:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif ROTATE_CAMERA == 270:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Lật gương
    if FLIP_HORIZONTAL:
        image = cv2.flip(image, 1)  # 1: Lật ngang (trái <-> phải)
    if FLIP_VERTICAL:
        image = cv2.flip(image, 0)  # 0: Lật dọc (trên <-> dưới)

    return image


def preprocess_anti_glare_clahe(
    image_bgr: np.ndarray,
    clip_limit: float = CLAHE_CLIP_LIMIT,
    tile_grid_size: tuple = CLAHE_GRID_SIZE
) -> np.ndarray:
    """
    Tiền xử lý giảm lóa sáng, khôi phục chi tiết khuôn mặt bằng CLAHE trên kênh L (LAB).
    - Chuyển BGR -> LAB để tách biệt kênh L (Lightness) và A, B (màu sắc).
    - Cân bằng histogram thích ứng cục bộ (CLAHE) trên kênh L để dập tắt lóa sáng.
    - Giữ nguyên 100% màu da tự nhiên ở kênh A và B.
    """
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr

    # Bước 1: Chuyển sang không gian màu LAB
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Bước 2: Áp dụng CLAHE chỉ trên kênh L
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l_channel)

    # Bước 3: Gộp lại và chuyển về BGR
    enhanced_lab = cv2.merge((l_clahe, a_channel, b_channel))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


def enhance_image_clarity(image_bgr: np.ndarray) -> np.ndarray:
    """
    Tăng cường độ nét (Unsharp Mask) và phục hồi vi mô chi tiết khuôn mặt (mắt, mũi, môi)
    khi người dùng đứng ở cự ly xa (1m - 2m) trên camera phân giải thấp (QVGA).
    """
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr

    # Unsharp Mask: Tăng cường chi tiết cạnh nhưng không gây nhiễu hạt
    gaussian = cv2.GaussianBlur(image_bgr, (0, 0), sigmaX=1.5)
    sharpened = cv2.addWeighted(image_bgr, 1.35, gaussian, -0.35, 0)
    return sharpened


def check_anti_spoofing(image: np.ndarray) -> tuple:
    """
    Kiểm tra tính chân thực của khuôn mặt (người thật vs ảnh giả mạo).
    Trả về: (is_real: bool, score: float)
    Nếu không có khuôn mặt trong ảnh, ném ValueError.
    """
    face_objs = DeepFace.extract_faces(
        img_path=image,
        detector_backend=DETECTOR_BACKEND,
        anti_spoofing=True,
        enforce_detection=True  # Ném ValueError nếu không có khuôn mặt
    )

    if face_objs and len(face_objs) > 0:
        face = face_objs[0]
        is_real = face.get("is_real", False)
        antispoof_score = face.get("antispoof_score", 0.0)
        return (is_real, antispoof_score)

    return (False, 0.0)


def process_face_recognition(image: np.ndarray, client: mqtt.Client):
    """
    Xử lý nhận diện với cơ chế đa tầng thích ứng ánh sáng:
    1. Ưu tiên quét trên ảnh gốc để giữ nguyên vẹn chất lượng da cho Anti-Spoofing.
    2. Nếu ảnh bị lóa sáng gắt không thấy mặt -> Tự động Fallback sang CLAHE.
    """
    start_time = time.time()

    try:
        spoof_score = 1.0
        face_img = image

        # ================================================================
        # BƯỚC 1: PHÁT HIỆN MẶT & KIỂM TRA CHỐNG GIẢ MẠO (ANTI-SPOOFING)
        # ================================================================
        if ANTI_SPOOFING:
            try:
                # Quét trên ảnh gốc trước để giữ nguyên vẹn kết cấu da tự nhiên
                is_real, spoof_score = check_anti_spoofing(image)
            except ValueError:
                # Nếu ảnh gốc bị lóa sáng nặng khiến YOLO không tìm thấy mặt
                if ENABLE_CLAHE:
                    # Kích hoạt Fallback: dùng ảnh cân bằng sáng CLAHE để cứu nguy
                    clahe_img = preprocess_anti_glare_clahe(image)
                    is_real, spoof_score = check_anti_spoofing(clahe_img)
                    face_img = clahe_img
                else:
                    raise

            if not is_real:
                # -------------------------------------------------------------
                # TRẠNG THÁI 2: "GIẢ MẠO" (ANTI-SPOOFING)
                # -------------------------------------------------------------
                elapsed = time.time() - start_time
                print("[AI] ⚠️  CẢNH BÁO: Phát hiện giả mạo (Ảnh/Màn hình)!")
                print(f"     🛡️  Anti-Spoof : FAKE (score: {spoof_score:.4f})")
                print(f"     ⏱️  Xử lý     : {elapsed:.4f}s")
                print("     🔒 Lệnh       : DENIED")
                print("-" * 55)
                client.publish(TOPIC_CONTROL, "DENIED")
                return

            # Người thật: in trạng thái xác minh Anti-Spoofing thành công
            print(f"[AI] 🛡️  Anti-Spoofing: REAL (score: {spoof_score:.4f})")

        # ================================================================
        # BƯỚC 2: SO KHỚP ĐẶC TRƯNG KHUÔN MẶT (FACENET512)
        # ================================================================
        # Tăng cường độ nét ảnh giúp mô hình nhận diện tốt hơn khi đứng xa
        enhanced_face_img = enhance_image_clarity(face_img)

        results = DeepFace.find(
            img_path=enhanced_face_img,
            db_path=FACES_DB_PATH,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            distance_metric=DISTANCE_METRIC,
            enforce_detection=False,
            similarity_search=True,
            silent=True
        )

        elapsed = time.time() - start_time

        if results and len(results) > 0 and not results[0].empty:
            best_match = results[0].iloc[0]

            # Lấy khoảng cách distance
            distance_col = f"{MODEL_NAME}_{DISTANCE_METRIC}"
            if distance_col not in results[0].columns and "distance" in results[0].columns:
                distance_col = "distance"
            distance = float(best_match[distance_col])

            # Lấy tên người dùng từ đường dẫn ảnh
            identity_path = str(best_match["identity"])
            user_name = os.path.basename(os.path.dirname(identity_path))
            if not user_name or user_name == os.path.basename(FACES_DB_PATH.rstrip("/\\")):
                user_name = os.path.splitext(os.path.basename(identity_path))[0]

            if distance <= DISTANCE_THRESHOLD:
                # -------------------------------------------------------------
                # TRẠNG THÁI 4: "CHẤP NHẬN" (CHỦ NHÀ)
                # -------------------------------------------------------------
                print("[AI] ✅ NHẬN DIỆN THÀNH CÔNG")
                print(f"     👤 Người dùng : {user_name}")
                print(f"     📐 Distance   : {distance:.6f} <= {DISTANCE_THRESHOLD}")
                print("     🛡️  Anti-Spoof : REAL")
                print(f"     ⏱️  Xử lý     : {elapsed:.4f}s")
                print("     🔓 Lệnh       : OPEN_FACE")
                print("-" * 55)
                client.publish(TOPIC_CONTROL, "OPEN_FACE")
            else:
                # -------------------------------------------------------------
                # TRẠNG THÁI 3: "TỪ CHỐI" (NGƯỜI LẠ - KHOẢNG CÁCH QUÁ XA)
                # -------------------------------------------------------------
                print("[AI] ❌ KẺ LẠ MẶT - TỪ CHỐI")
                print(f"     👤 Khớp nhất  : {user_name}")
                print(f"     📐 Distance   : {distance:.6f} > {DISTANCE_THRESHOLD}")
                print("     🛡️  Anti-Spoof : REAL")
                print(f"     ⏱️  Xử lý     : {elapsed:.4f}s")
                print("     🔒 Lệnh       : DENIED")
                print("-" * 55)
                client.publish(TOPIC_CONTROL, "DENIED")
        else:
            # -----------------------------------------------------------------
            # TRẠNG THÁI 3: "TỪ CHỐI" (NGƯỜI LẠ - KHÔNG KHỚP DỮ LIỆU)
            # -----------------------------------------------------------------
            print("[AI] ❌ KẺ LẠ MẶT - TỪ CHỐI")
            print("     👤 Người dùng : Không xác định (Người lạ)")
            print("     📐 Distance   : N/A (không khớp dữ liệu)")
            print("     🛡️  Anti-Spoof : REAL")
            print(f"     ⏱️  Xử lý     : {elapsed:.4f}s")
            print("     🔒 Lệnh       : DENIED")
            print("-" * 55)
            client.publish(TOPIC_CONTROL, "DENIED")

    except ValueError:
        # ---------------------------------------------------------------------
        # TRẠNG THÁI 1: "CHỜ NGƯỜI DÙNG" (KHÔNG TÌM THẤY KHUÔN MẶT)
        # ---------------------------------------------------------------------
        # Chỉ in log chờ người dùng, TUYỆT ĐỐI KHÔNG publish lệnh MQTT nào
        print("[AI] ⏳ Đang chờ người dùng đứng vào camera...")
        print("-" * 55)
        client.publish(TOPIC_CONTROL, "NO_FACE")

    except Exception as e:
        print(f"[AI] ⚠️  Lỗi xử lý: {e}")
        print("-" * 55)


def process_frames(client: mqtt.Client):
    """
    Worker Thread chạy ngầm: Lấy ảnh từ queue, điều chỉnh chiều và tiến hành nhận diện.
    Hỗ trợ phím tắt điều chỉnh trực tiếp trên cửa sổ camera:
      - 'r': Đổi góc xoay (180° -> 0° -> 90° -> 270°)
      - 'f': Bật/Tắt lật ngang (Trái <-> Phải)
      - 'v': Bật/Tắt lật dọc (Trên <-> Dưới)
    """
    global ROTATE_CAMERA, FLIP_HORIZONTAL, FLIP_VERTICAL, LAST_AI_SCAN
    print("[THREAD] 🔄 Luồng xử lý AI đã sẵn sàng.")

    while True:
        try:
            raw_image = frame_queue.get(timeout=1.0)

            # Điều chỉnh chiều xoay / lật camera (khắc phục camera bị ngược)
            image = orient_image(raw_image)

            # Hiển thị luồng video lên cửa sổ Live Camera
            cv2.imshow("ESP32-S3 Live Camera", image)

            # Bắt phím điều chỉnh khi người dùng thao tác trên cửa sổ
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('r'), ord('R')):
                rot_cycle = {180: 0, 0: 90, 90: 270, 270: 180}
                ROTATE_CAMERA = rot_cycle.get(ROTATE_CAMERA, 180)
                print(f"[CAM] 🔄 Phím 'r': Đổi góc xoay -> {ROTATE_CAMERA}°")
            elif key in (ord('f'), ord('F')):
                FLIP_HORIZONTAL = not FLIP_HORIZONTAL
                print(f"[CAM] 🔄 Phím 'f': Lật ngang (Trái <-> Phải) -> {'BẬT' if FLIP_HORIZONTAL else 'TẮT'}")
            elif key in (ord('v'), ord('V')):
                FLIP_VERTICAL = not FLIP_VERTICAL
                print(f"[CAM] 🔄 Phím 'v': Lật dọc (Trên <-> Dưới) -> {'BẬT' if FLIP_VERTICAL else 'TẮT'}")
            elif key in (ord('s'), ord('S')):
                # Lưu trực tiếp ảnh mẫu chụp từ camera tại cự ly này vào faces_db/Dinh Cong Son
                target_user = "Dinh Cong Son"
                user_folder = os.path.join(FACES_DB_PATH, target_user)
                if not os.path.exists(user_folder):
                    os.makedirs(user_folder, exist_ok=True)

                filename = f"esp32_dist_{int(time.time())}.jpg"
                filepath = os.path.join(user_folder, filename)
                cv2.imwrite(filepath, image)

                # Xóa file cache .pkl để DeepFace tự động nạp lại ảnh mới
                for f in os.listdir(FACES_DB_PATH):
                    if f.endswith(".pkl"):
                        try:
                            os.remove(os.path.join(FACES_DB_PATH, f))
                        except Exception:
                            pass
                print(f"[DB] 📸 Đã lưu ảnh mẫu cự ly này: {filepath}")
                print("[DB] 🔄 Đã làm mới cơ sở dữ liệu khuôn mặt! Lần quét tới sẽ nhận diện ngay.")

            current_time = time.time()
            if current_time - LAST_AI_SCAN >= 4.0:
                print(f"[AI] 🖼️  Kích thước ảnh: {image.shape[1]}x{image.shape[0]}")
                process_face_recognition(image, client)
                LAST_AI_SCAN = time.time()

        except queue.Empty:
            cv2.waitKey(1)
            continue
        except Exception as e:
            print(f"[THREAD] ❌ Lỗi luồng: {e}")


def create_mqtt_client() -> mqtt.Client:
    """
    Khởi tạo MQTT Client với TLS và cơ chế xử lý tin nhắn non-blocking.
    """
    client = mqtt.Client(
        client_id="AI_Server_RTX4050",
        protocol=mqtt.MQTTv5
    )

    client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)

    client.tls_set(
        ca_certs=None,
        certfile=None,
        keyfile=None,
        cert_reqs=ssl.CERT_REQUIRED,
        tls_version=ssl.PROTOCOL_TLSv1_2
    )

    def on_connect(client, userdata, flags, rc, properties=None):
        if rc == 0:
            print("[MQTT] ✅ Kết nối HiveMQ Cloud thành công!")
            client.subscribe(TOPIC_CAMERA)
            print(f"[MQTT] 📡 Đã subscribe: {TOPIC_CAMERA}")
            print("[AI] 🔍 Hệ thống sẵn sàng nhận diện...\n" + "=" * 55)
            client.publish(TOPIC_CONTROL, "READY")
        else:
            print(f"[MQTT] ❌ Kết nối thất bại, rc: {rc}")

    def on_message(client, userdata, msg):
        try:
            size_kb = len(msg.payload) / 1024
            print(f"\n[MQTT] 📩 Nhận ảnh ({size_kb:.1f} KB)")

            b64_payload = msg.payload.decode('utf-8')
            image = decode_base64_to_image(b64_payload)

            if image is None or image.size == 0:
                return

            # Cơ chế Queue maxsize=1: nếu đầy thì bỏ frame cũ, nạp frame mới nhất
            try:
                frame_queue.put_nowait(image)
            except queue.Full:
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass
                frame_queue.put_nowait(image)

        except Exception as e:
            print(f"[AI] ❌ Lỗi giải mã message: {e}")

    client.on_connect = on_connect
    client.on_message = on_message

    return client


# ====================================================================
# ĐIỂM VÀO CHƯƠNG TRÌNH
# ====================================================================
if __name__ == "__main__":
    print("=" * 55)
    print("  AI SERVER – Hệ thống Khóa cửa Thông minh")
    print("  Mô hình: Facenet512 | Detector: YOLOv8n")
    print("  Bảo vệ: Anti-Spoofing (FASNet)")
    print(f"  Tiền xử lý CLAHE: {'BẬT ☀️' if ENABLE_CLAHE else 'TẮT'}")
    print(f"  Định hướng Camera: Xoay {ROTATE_CAMERA}° | Lật ngang: {'BẬT' if FLIP_HORIZONTAL else 'TẮT'} | Lật dọc: {'BẬT' if FLIP_VERTICAL else 'TẮT'}")
    print("  (Phím tắt trên cửa sổ Cam: 'r': Xoay | 'f': Lật ngang | 'v': Lật dọc)")
    print("=" * 55)

    if not os.path.exists(FACES_DB_PATH):
        os.makedirs(FACES_DB_PATH)
        print(f"[SYS] 📁 Đã tạo thư mục: {FACES_DB_PATH}")

    # Bước 1: Khởi động nóng mô hình
    warmup_model()
    print("-" * 55)

    # Bước 2: Khởi tạo kết nối MQTT
    mqtt_client = create_mqtt_client()
    mqtt_client.connect(MQTT_BROKER, MQTT_PORT)
    print("[MQTT] 🔄 Đang kết nối HiveMQ Cloud...")

    # Bước 3: Khởi động Thread xử lý AI
    ai_thread = threading.Thread(
        target=process_frames,
        args=(mqtt_client,),
        daemon=True,
        name="AI_Processing_Thread"
    )
    ai_thread.start()

    try:
        mqtt_client.loop_forever()
    except KeyboardInterrupt:
        print("\n[SYS] 🛑 Nhận tín hiệu ngắt (Ctrl+C).")
    finally:
        mqtt_client.disconnect()
        cv2.destroyAllWindows()
        print("[MQTT] 🔌 Đã ngắt kết nối MQTT.")
        print("[SYS] 👋 Server AI đã dừng.")
