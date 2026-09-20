"""
╔══════════════════════════════════════════════════════════════════════╗
║  ai_server.py – Trạm AI Nhận diện Khuôn mặt Trung tâm            ║
║  Tác giả : Kỹ sư Hệ thống Nhúng & AI                             ║
║  Mô tả  : Subscribe ảnh từ MQTT, nhận diện bằng DeepFace          ║
║            (Facenet512 + GPU RTX 4050), publish lệnh điều khiển.   ║
║  Tối ưu : Threading + Queue tách biệt luồng nhận MQTT và xử lý   ║
║            AI, tránh nghẽn cổ chai (blocking) trong on_message.    ║
║  Bảo mật: Anti-Spoofing (FASNet) phát hiện ảnh giả mạo từ màn    ║
║            hình điện thoại hoặc ảnh in trước khi nhận diện.       ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os

# ====================================================================
# CẤU HÌNH BACKEND KERAS → PYTORCH (QUAN TRỌNG: PHẢI ĐẶT TRƯỚC IMPORT)
# ====================================================================
# Keras 3.x hỗ trợ đa backend: tensorflow, torch, jax.
# TensorFlow >= 2.11 KHÔNG hỗ trợ GPU trên Windows native.
# → Ép Keras dùng PyTorch làm backend để tận dụng CUDA trên RTX 4050.
# Biến môi trường PHẢI được set TRƯỚC khi import keras/deepface,
# vì Keras đọc biến này 1 lần duy nhất lúc khởi tạo module.
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
# HẰNG SỐ CẤU HÌNH – Tự động tải từ file .env
# ====================================================================
MQTT_BROKER   = os.getenv("MQTT_BROKER")
MQTT_PORT     = int(os.getenv("MQTT_PORT", 8883))
MQTT_USERNAME = os.getenv("MQTT_USERNAME")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD")

# Topic MQTT
TOPIC_CAMERA  = "haui/smartdoor/camera"        # Nhận ảnh từ camera
TOPIC_CONTROL = "haui/smartdoor/control"        # Gửi lệnh điều khiển

# Cấu hình AI
MODEL_NAME       = "Facenet512"    # Mô hình nhận diện (512-d embedding)
DISTANCE_METRIC  = "cosine"        # Metric đo khoảng cách vector đặc trưng
DETECTOR_BACKEND = "yolov8n"       # Bộ phát hiện khuôn mặt (YOLOv8 nano - cực nhanh)
FACES_DB_PATH    = "faces_db/"     # Thư mục chứa ảnh khuôn mặt đã đăng ký
ANTI_SPOOFING    = True            # Bật chống giả mạo khuôn mặt (FASNet)

# Ngưỡng khoảng cách cosine – giá trị càng nhỏ càng giống
# Facenet512 + cosine: ngưỡng mặc định ~0.30
# Bạn có thể tinh chỉnh sau khi quan sát distance trên terminal
DISTANCE_THRESHOLD = 0.65

# ====================================================================
# HÀNG ĐỢI KHUNG HÌNH (Frame Queue)
# ====================================================================
# maxsize=1: chỉ giữ DUY NHẤT 1 ảnh mới nhất trong hàng đợi.
# Nguyên lý thời gian thực: nếu AI xử lý chậm hơn tốc độ camera gửi,
# ta BỎ ảnh cũ và chỉ giữ ảnh mới nhất → đảm bảo luôn nhận diện
# khuôn mặt ở thời điểm gần nhất, không xử lý ảnh quá khứ.
frame_queue = queue.Queue(maxsize=1)


def warmup_model():
    """
    Khởi tạo "nóng" mô hình AI bằng ảnh dummy (ảnh đen).

    Nguyên lý:
    - DeepFace sử dụng lazy loading: mô hình chỉ được tải khi gọi
      hàm nhận diện lần đầu tiên, gây độ trễ lớn (3-8 giây).
    - Bằng cách chạy DeepFace.represent() với ảnh dummy ngay lúc
      khởi động, ta ÉP tải toàn bộ trọng số (weights) của Facenet512
      vào VRAM của GPU (RTX 4050) trước.
    - Đồng thời warmup mô hình Anti-Spoofing (FASNet) bằng cách gọi
      DeepFace.extract_faces() với anti_spoofing=True, ép tải trọng số
      FASNet vào bộ nhớ sẵn để tránh độ trễ lần đầu.
    - Các lần nhận diện tiếp theo sẽ đạt tốc độ thời gian thực
      vì mô hình đã sẵn sàng trong bộ nhớ GPU.
    """
    # === KIỂM TRA GPU TRƯỚC KHI TẢI MÔ HÌNH ===
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[GPU] ✅ CUDA sẵn sàng: {gpu_name} ({vram_gb:.1f} GB VRAM)")
        print(f"[GPU] 🔧 PyTorch {torch.__version__} | Keras backend: torch")
    else:
        print("[GPU] ⚠️  CUDA KHÔNG khả dụng! Đang chạy trên CPU (chậm).")

    print("[AI] 🧠 Đang tải mô hình Facenet512 vào GPU...")
    start = time.time()

    # Tạo ảnh đen 224x224x3 (kích thước input chuẩn của Facenet512)
    # numpy.zeros tạo mảng toàn số 0 → ảnh đen hoàn toàn
    dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)

    try:
        # Gọi represent() để ép tải mô hình nhận diện vào bộ nhớ
        # enforce_detection=False: không báo lỗi nếu không tìm thấy mặt
        DeepFace.represent(
            img_path=dummy_img,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False
        )
        elapsed = time.time() - start
        print(f"[AI] ✅ Mô hình Facenet512 đã sẵn sàng! ({elapsed:.2f}s)")
    except Exception as e:
        print(f"[AI] ⚠️  Cảnh báo khi warmup Facenet512: {e}")
        print("[AI] 🔄 Mô hình vẫn có thể hoạt động bình thường.")

    # === WARMUP MÔ HÌNH ANTI-SPOOFING (FASNet) ===
    # FASNet (Face Anti-Spoofing Network) phân biệt khuôn mặt thật/giả.
    # Tải trước để tránh độ trễ lớn khi xử lý frame thực đầu tiên.
    if ANTI_SPOOFING:
        print("[AI] 🛡️  Đang tải mô hình Anti-Spoofing (FASNet)...")
        start_spoof = time.time()
        try:
            # extract_faces() với anti_spoofing=True sẽ ép tải FASNet
            DeepFace.extract_faces(
                img_path=dummy_img,
                detector_backend=DETECTOR_BACKEND,
                anti_spoofing=True,
                enforce_detection=False
            )
            elapsed_spoof = time.time() - start_spoof
            print(f"[AI] ✅ Mô hình Anti-Spoofing đã sẵn sàng! ({elapsed_spoof:.2f}s)")
        except Exception as e:
            # FASNet có thể lỗi khi tải lần đầu với ảnh dummy (không có mặt)
            # Điều này bình thường – mô hình vẫn được cache vào bộ nhớ
            print(f"[AI] ⚠️  Cảnh báo khi warmup Anti-Spoofing: {e}")
            print("[AI] 🔄 Mô hình Anti-Spoofing sẽ tải khi gặp khuôn mặt thực.")


def decode_base64_to_image(b64_string: str) -> np.ndarray:
    """
    Giải mã chuỗi Base64 ngược về ảnh NumPy array (định dạng OpenCV).

    Nguyên lý:
    1. base64.b64decode: chuyển chuỗi ASCII Base64 → bytes JPEG gốc.
    2. np.frombuffer: chuyển bytes → mảng NumPy 1 chiều (dtype uint8).
    3. cv2.imdecode: giải nén JPEG → mảng NumPy 3 chiều (H x W x 3)
       ở định dạng BGR mà OpenCV sử dụng.

    Quy trình ngược lại hoàn toàn so với bước mã hoá trong mock_camera.py.
    """
    jpeg_bytes = base64.b64decode(b64_string)
    np_array = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    return image


def check_anti_spoofing(image: np.ndarray) -> tuple:
    """
    Kiểm tra khuôn mặt có phải người thật hay ảnh giả mạo.

    Nguyên lý hoạt động của FASNet (Face Anti-Spoofing Network):
    - FASNet là mạng neural chuyên biệt, được huấn luyện để phân biệt
      khuôn mặt thật (live/real) và khuôn mặt giả (spoof/fake).
    - Mô hình phân tích các đặc trưng vi mô (micro-texture) mà mắt
      thường không thể nhận ra: hiệu ứng moiré từ màn hình, phản xạ
      ánh sáng bất thường, thiếu chiều sâu 3D của ảnh in/ảnh điện thoại.
    - DeepFace.extract_faces(anti_spoofing=True) trả về dict chứa:
      + is_real (bool): True nếu là người thật, False nếu là ảnh giả.
      + antispoof_score (float): Điểm số 0.0-1.0, càng cao càng "thật".

    Returns:
        tuple: (is_real: bool, spoof_score: float)
        - is_real = True → Khuôn mặt người thật
        - is_real = False → Ảnh giả mạo (từ màn hình/ảnh in)
        - spoof_score: Điểm anti-spoof (0.0 = chắc chắn giả, 1.0 = chắc chắn thật)
    """
    try:
        # DeepFace.extract_faces() với anti_spoofing=True sẽ chạy FASNet
        # song song với detector để đánh giá tính chân thực của khuôn mặt
        face_objs = DeepFace.extract_faces(
            img_path=image,
            detector_backend=DETECTOR_BACKEND,
            anti_spoofing=True,
            enforce_detection=False
        )

        # Kiểm tra có phát hiện khuôn mặt nào không
        if face_objs and len(face_objs) > 0:
            face = face_objs[0]
            is_real = face.get("is_real", False)
            antispoof_score = face.get("antispoof_score", 0.0)
            return (is_real, antispoof_score)
        else:
            # Không phát hiện khuôn mặt → coi như không hợp lệ
            return (False, 0.0)

    except Exception as e:
        # Lỗi khi chạy FASNet (lỗi tải model, ảnh lỗi, v.v.)
        # An toàn: từ chối mở cửa khi không xác minh được
        print(f"[AI] ⚠️  Lỗi Anti-Spoofing: {e}")
        print("[AI] 🛡️  An toàn mặc định: coi như ảnh giả mạo.")
        return (False, 0.0)


def process_face_recognition(image: np.ndarray, client: mqtt.Client):
    """
    Thực hiện nhận diện khuôn mặt với 2 lớp bảo vệ và publish lệnh điều khiển.

    Luồng quyết định 2 lớp:
    ┌─────────────────────────────────────────────────────┐
    │ LỚP 1: Anti-Spoofing (FASNet)                      │
    │ → Kiểm tra khuôn mặt thật hay giả mạo              │
    │ → Nếu FAKE → DENIED ngay lập tức (không cần LỚP 2) │
    ├─────────────────────────────────────────────────────┤
    │ LỚP 2: Face Recognition (Facenet512)                │
    │ → Chỉ chạy khi LỚP 1 xác nhận REAL                 │
    │ → So sánh distance với faces_db/                    │
    │ → distance < threshold → OPEN_FACE                  │
    │ → distance >= threshold → DENIED                    │
    └─────────────────────────────────────────────────────┘

    Thuật toán:
    1. [LỚP 1] DeepFace.extract_faces(anti_spoofing=True) kiểm tra tính
       chân thực của khuôn mặt bằng mô hình FASNet.
    2. [LỚP 2] DeepFace.find() trích xuất vector đặc trưng 512 chiều và
       so sánh với faces_db/ bằng khoảng cách cosine.
    3. Chỉ khi CẢ HAI lớp đều PASS → publish "OPEN_FACE".
    """
    start_time = time.time()

    try:
        # ================================================================
        # LỚP 1: KIỂM TRA CHỐNG GIẢ MẠO (ANTI-SPOOFING)
        # ================================================================
        # Chạy FASNet TRƯỚC khi nhận diện để tiết kiệm tài nguyên GPU.
        # Nếu ảnh giả → từ chối ngay, không cần tốn thời gian so sánh
        # khuôn mặt với toàn bộ cơ sở dữ liệu faces_db/.
        if ANTI_SPOOFING:
            is_real, spoof_score = check_anti_spoofing(image)

            if not is_real:
                elapsed = time.time() - start_time
                # 🚨 PHÁT HIỆN GIẢ MẠO → TỪ CHỐI NGAY LẬP TỨC
                print(f"[AI] 🚨 CẢNH BÁO: Phát hiện khuôn mặt GIẢ MẠO!")
                print(f"     🛡️  Spoof Score: {spoof_score:.4f} (thấp = giả)")
                print(f"     📋 Loại tấn công có thể: ảnh điện thoại / ảnh in")
                print(f"     ⏱️  Xử lý     : {elapsed:.4f}s")
                print(f"     🚫 Lệnh       : DENIED")
                print("-" * 55)

                client.publish(TOPIC_CONTROL, "DENIED")
                return  # Dừng ngay, KHÔNG chạy LỚP 2

            # Khuôn mặt THẬT → tiếp tục LỚP 2
            print(f"[AI] 🛡️  Anti-Spoofing: REAL (score: {spoof_score:.4f})")

        # ================================================================
        # LỚP 2: NHẬN DIỆN KHUÔN MẶT (FACE RECOGNITION)
        # ================================================================
        # Chỉ chạy khi LỚP 1 xác nhận khuôn mặt THẬT (hoặc Anti-Spoofing tắt).
        # DeepFace.find() trả về list các DataFrame, mỗi DataFrame
        # chứa kết quả khớp cho 1 khuôn mặt phát hiện được trong ảnh.
        # - db_path: thư mục chứa ảnh đã đăng ký (1 ảnh/người hoặc nhiều)
        # - model_name: kiến trúc mạng neural trích xuất đặc trưng
        # - distance_metric: phương pháp đo khoảng cách giữa 2 vector
        # - enforce_detection=False: vẫn xử lý ngay cả khi không phát
        #   hiện rõ khuôn mặt (tránh crash trong điều kiện ánh sáng kém)
        # - silent=True: tắt log thừa của DeepFace
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

        # === PHÂN TÍCH KẾT QUẢ ===
        # results là list của DataFrame. Kiểm tra DataFrame đầu tiên
        # (khuôn mặt chính trong ảnh) có dữ liệu hay không.
        if results and len(results) > 0 and not results[0].empty:
            # Lấy kết quả khớp tốt nhất (dòng đầu tiên, distance nhỏ nhất)
            best_match = results[0].iloc[0]

            # Trích xuất tên cột chứa distance
            # DeepFace v0.0.101+ dùng cột "distance" thay vì "{model}_{metric}"
            # Tự động phát hiện tên cột để tương thích mọi phiên bản
            df = results[0]
            distance_col = f"{MODEL_NAME}_{DISTANCE_METRIC}"
            if distance_col not in df.columns and "distance" in df.columns:
                distance_col = "distance"
            distance = best_match[distance_col]

            # Trích xuất đường dẫn ảnh khớp → lấy tên người dùng
            # Quy ước: faces_db/TenNguoi/anh.jpg → tên = "TenNguoi"
            identity_path = best_match["identity"]
            # Lấy tên thư mục cha của file ảnh làm tên người dùng
            user_name = os.path.basename(os.path.dirname(identity_path))
            if not user_name:
                # Trường hợp ảnh nằm trực tiếp trong faces_db/
                user_name = os.path.splitext(os.path.basename(identity_path))[0]

            # === LUỒNG QUYẾT ĐỊNH LỚP 2 ===
            if distance < DISTANCE_THRESHOLD:
                # ✅ KHỚP + NGƯỜI THẬT → Mở cửa
                print(f"[AI] ✅ NHẬN DIỆN THÀNH CÔNG")
                print(f"     👤 Người dùng : {user_name}")
                print(f"     📏 Distance   : {distance:.6f}")
                print(f"     🛡️  Anti-Spoof : REAL")
                print(f"     ⏱️  Xử lý     : {elapsed:.4f}s")
                print(f"     🔓 Lệnh       : OPEN_FACE")
                print("-" * 55)

                # Publish lệnh mở khóa cửa
                client.publish(TOPIC_CONTROL, "OPEN_FACE")
            else:
                # ❌ Distance quá cao – người lạ (nhưng là người thật)
                print(f"[AI] ⚠️  DISTANCE VƯỢT NGƯỠNG")
                print(f"     📏 Distance   : {distance:.6f} (ngưỡng: {DISTANCE_THRESHOLD})")
                print(f"     🛡️  Anti-Spoof : REAL")
                print(f"     ⏱️  Xử lý     : {elapsed:.4f}s")
                print(f"     🚫 Lệnh       : DENIED")
                print("-" * 55)

                client.publish(TOPIC_CONTROL, "DENIED")
        else:
            # === KHÔNG TÌM THẤY KHUÔN MẶT KHỚP ===
            print(f"[AI] 🚨 NGƯỜI LẠ - TỪ CHỐI")
            print(f"     📏 Distance   : N/A (không có dữ liệu khớp)")
            print(f"     ⏱️  Xử lý     : {elapsed:.4f}s")
            print(f"     🚫 Lệnh       : DENIED")
            print("-" * 55)

            client.publish(TOPIC_CONTROL, "DENIED")

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[AI] ❌ Lỗi nhận diện: {e}")
        print(f"     ⏱️  Xử lý     : {elapsed:.4f}s")
        print("-" * 55)

        client.publish(TOPIC_CONTROL, "DENIED")


def process_frames(client: mqtt.Client):
    """
    Hàm xử lý AI chạy trên Daemon Thread riêng biệt.

    Nguyên lý kiến trúc Producer-Consumer:
    - Producer (on_message): nhận ảnh từ MQTT → đẩy vào frame_queue.
    - Consumer (hàm này): lấy ảnh từ frame_queue → chạy DeepFace.

    Tách biệt 2 luồng giải quyết nghẽn cổ chai:
    - Luồng MQTT (producer) luôn sẵn sàng nhận message mới, không bị
      chặn bởi thời gian xử lý AI (3-5 giây/frame trên GPU).
    - Luồng AI (consumer) xử lý tuần tự từng ảnh, đảm bảo GPU không
      bị quá tải bởi nhiều request đồng thời.

    frame_queue.get(timeout=1.0):
    - timeout=1.0 giúp thread không bị treo vĩnh viễn khi queue rỗng,
      cho phép kiểm tra điều kiện dừng (daemon thread tự kết thúc
      khi main thread kết thúc).
    """
    print("[THREAD] 🔄 Luồng xử lý AI đã khởi động.")

    while True:
        try:
            # Lấy ảnh từ hàng đợi – chờ tối đa 1 giây
            # Nếu hết timeout mà queue vẫn rỗng → quay lại vòng lặp
            image = frame_queue.get(timeout=1.0)

            print(f"[AI] 🖼️  Kích thước ảnh: {image.shape[1]}x{image.shape[0]}")

            # Chạy nhận diện khuôn mặt trên thread riêng
            # Không ảnh hưởng tới luồng nhận MQTT
            process_face_recognition(image, client)

        except queue.Empty:
            # Queue rỗng sau 1 giây chờ – không có ảnh mới
            # Tiếp tục vòng lặp, chờ ảnh tiếp theo
            continue

        except Exception as e:
            print(f"[THREAD] ❌ Lỗi trong luồng xử lý: {e}")


def create_mqtt_client() -> mqtt.Client:
    """
    Khởi tạo MQTT client với TLS và các callback xử lý sự kiện.

    Nguyên lý:
    - on_connect: Khi kết nối thành công, tự động subscribe topic camera.
      Đặt subscribe trong on_connect đảm bảo tự động re-subscribe khi
      mất kết nối và kết nối lại (auto-reconnect).
    - on_message: Callback chỉ giải mã ảnh và đẩy vào queue, KHÔNG gọi
      DeepFace trực tiếp. Nhờ vậy callback kết thúc ngay lập tức (~1ms),
      giải phóng luồng MQTT để nhận message tiếp theo.
    """
    client = mqtt.Client(
        client_id="AI_Server_RTX4050",
        protocol=mqtt.MQTTv5
    )

    client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)

    # Cấu hình TLS bảo mật
    client.tls_set(
        ca_certs=None,
        certfile=None,
        keyfile=None,
        cert_reqs=ssl.CERT_REQUIRED,
        tls_version=ssl.PROTOCOL_TLSv1_2
    )

    # === CALLBACK KẾT NỐI ===
    def on_connect(client, userdata, flags, rc, properties=None):
        if rc == 0:
            print("[MQTT] ✅ Kết nối HiveMQ Cloud thành công!")
            # Subscribe topic camera ngay khi kết nối
            client.subscribe(TOPIC_CAMERA)
            print(f"[MQTT] 📡 Đã subscribe: {TOPIC_CAMERA}")
            print("[AI] 🔍 Đang chờ ảnh từ camera...\n" + "=" * 55)
        else:
            print(f"[MQTT] ❌ Kết nối thất bại, mã lỗi: {rc}")

    # === CALLBACK NHẬN MESSAGE (PRODUCER) ===
    def on_message(client, userdata, msg):
        """
        Nhận ảnh Base64 từ MQTT → giải mã → đẩy vào frame_queue.

        QUAN TRỌNG: Hàm này KHÔNG gọi DeepFace trực tiếp.
        - Paho MQTT gọi on_message trên luồng network duy nhất.
        - Nếu on_message bị chặn bởi AI (3-5s), toàn bộ việc nhận
          message MQTT sẽ bị đình trệ → mất ảnh, mất heartbeat.
        - Bằng cách chỉ đẩy ảnh vào queue (~1ms), luồng MQTT được
          giải phóng ngay lập tức để nhận message tiếp theo.

        Cơ chế xử lý queue đầy (maxsize=1):
        - Nếu AI đang xử lý chậm và queue đã có 1 ảnh cũ,
          ta XÓA ảnh cũ (queue.get_nowait) và NẠP ảnh mới nhất.
        - Đảm bảo AI luôn xử lý frame MỚI NHẤT, không xử lý
          ảnh quá khứ đã lỗi thời (nguyên tắc thời gian thực).
        """
        print(f"\n[MQTT] 📩 Nhận ảnh ({len(msg.payload) / 1024:.1f} KB)")

        try:
            # Bước 1: Giải mã Base64 → ảnh OpenCV (NumPy array)
            b64_payload = msg.payload.decode('utf-8')
            image = decode_base64_to_image(b64_payload)

            # Bước 2: Kiểm tra ảnh hợp lệ
            if image is None or image.size == 0:
                print("[AI] ⚠️  Ảnh giải mã không hợp lệ, bỏ qua.")
                return

            # Bước 3: Đẩy ảnh vào hàng đợi (non-blocking)
            # Nếu queue đầy → xóa ảnh cũ, nạp ảnh mới nhất
            try:
                frame_queue.put_nowait(image)
            except queue.Full:
                # Queue đầy: lấy ảnh cũ ra (bỏ đi) rồi nạp ảnh mới
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass
                frame_queue.put_nowait(image)
                print("[QUEUE] 🔄 Thay thế ảnh cũ bằng ảnh mới nhất.")

        except Exception as e:
            print(f"[AI] ❌ Lỗi xử lý message: {e}")

    client.on_connect = on_connect
    client.on_message = on_message

    return client


# ====================================================================
# ĐIỂM VÀO CHƯƠNG TRÌNH
# ====================================================================
if __name__ == "__main__":
    print("=" * 55)
    print("  AI SERVER – Hệ thống Khóa cửa Thông minh")
    print("  GPU: NVIDIA RTX 4050 | Model: Facenet512")
    print("  Kiến trúc: Threading + Queue (Non-blocking)")
    print(f"  Anti-Spoofing: {'BẬT 🛡️' if ANTI_SPOOFING else 'TẮT ⚠️'}")
    print("=" * 55)

    # Kiểm tra thư mục faces_db/ tồn tại
    if not os.path.exists(FACES_DB_PATH):
        os.makedirs(FACES_DB_PATH)
        print(f"[SYS] 📁 Đã tạo thư mục: {FACES_DB_PATH}")
        print(f"[SYS] ⚠️  Hãy thêm ảnh khuôn mặt vào {FACES_DB_PATH}")
        print(f"[SYS]    Cấu trúc: {FACES_DB_PATH}<TenNguoi>/anh1.jpg")
    else:
        # Đếm số ảnh đã đăng ký
        total_images = sum(
            len(files)
            for _, _, files in os.walk(FACES_DB_PATH)
            if files
        )
        print(f"[SYS] 📁 Thư mục faces_db/ chứa {total_images} ảnh đã đăng ký")

    print("-" * 55)

    # === BƯỚC 1: KHỞI TẠO NÓNG MÔ HÌNH AI ===
    # Ép tải trọng số Facenet512 + FASNet vào VRAM trước khi nhận ảnh thực
    warmup_model()

    print("-" * 55)

    # === BƯỚC 2: KHỞI TẠO KẾT NỐI MQTT ===
    mqtt_client = create_mqtt_client()
    mqtt_client.connect(MQTT_BROKER, MQTT_PORT)

    print("[MQTT] 🔄 Đang kết nối tới HiveMQ Cloud...")

    # === BƯỚC 3: KHỞI ĐỘNG LUỒNG XỬ LÝ AI (DAEMON THREAD) ===
    # daemon=True: thread tự động kết thúc khi main thread kết thúc
    # (khi người dùng nhấn Ctrl+C), không cần gọi thread.join().
    # Thread này chạy hàm process_frames() liên tục lấy ảnh từ
    # frame_queue và xử lý nhận diện bằng DeepFace.
    ai_thread = threading.Thread(
        target=process_frames,
        args=(mqtt_client,),
        daemon=True,
        name="AI_Processing_Thread"
    )
    ai_thread.start()

    try:
        # loop_forever() chạy vòng lặp mạng MQTT vĩnh viễn
        # Tự động nhận message và gọi callback on_message
        # on_message chỉ đẩy ảnh vào queue (~1ms) → không bị chặn
        # Chặn main thread cho đến khi nhận Ctrl+C
        mqtt_client.loop_forever()

    except KeyboardInterrupt:
        print("\n[SYS] 🛑 Nhận tín hiệu ngắt (Ctrl+C).")

    finally:
        mqtt_client.disconnect()
        print("[MQTT] 🔌 Đã ngắt kết nối MQTT.")
        print("[SYS] 👋 Server AI đã dừng.")
