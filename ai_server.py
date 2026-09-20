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

# Ngưỡng khoảng cách an toàn cho Facenet512 + Cosine (chuẩn an toàn: 0.30)
DISTANCE_THRESHOLD = 0.30

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
    Xử lý nhận diện với định dạng in log chuyên nghiệp:
    1. "Chờ người dùng" : Không phát hiện mặt -> Chờ, TUYỆT ĐỐI KHÔNG publish MQTT.
    2. "Giả mạo"        : Phát hiện giả mạo   -> Cảnh báo FAKE, publish "DENIED".
    3. "Từ chối"        : Người lạ / xa ngưỡng -> Từ chối, publish "DENIED".
    4. "Chấp nhận"      : Chủ nhà hợp lệ     -> Thành công, publish "OPEN_FACE".
    """
    start_time = time.time()

    try:
        spoof_score = 1.0
        # ================================================================
        # BƯỚC 1: PHÁT HIỆN MẶT & KIỂM TRA CHỐNG GIẢ MẠO (ANTI-SPOOFING)
        # ================================================================
        if ANTI_SPOOFING:
            is_real, spoof_score = check_anti_spoofing(image)

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
        results = DeepFace.find(
            img_path=image,
            db_path=FACES_DB_PATH,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            distance_metric=DISTANCE_METRIC,
            enforce_detection=False,
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
                print(f"     📐 Distance   : {distance:.6f}")
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
                print("     👤 Người dùng : Không xác định (Người lạ)")
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

    except Exception as e:
        print(f"[AI] ⚠️  Lỗi xử lý: {e}")
        print("-" * 55)


def process_frames(client: mqtt.Client):
    """
    Worker Thread chạy ngầm: Lấy ảnh từ queue và tiến hành nhận diện.
    """
    print("[THREAD] 🔄 Luồng xử lý AI đã sẵn sàng.")

    while True:
        try:
            image = frame_queue.get(timeout=1.0)
            print(f"[AI] 🖼️  Kích thước ảnh: {image.shape[1]}x{image.shape[0]}")
            process_face_recognition(image, client)

        except queue.Empty:
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
        print("[MQTT] 🔌 Đã ngắt kết nối MQTT.")
        print("[SYS] 👋 Server AI đã dừng.")
