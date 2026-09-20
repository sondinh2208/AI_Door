"""
╔══════════════════════════════════════════════════════════════════════╗
║  mock_camera.py – Giả lập ESP32-CAM truyền ảnh qua MQTT           ║
║  Tác giả : Kỹ sư Hệ thống Nhúng & AI                             ║
║  Mô tả  : Đọc khung hình từ Webcam, nén JPEG, mã hoá Base64      ║
║            rồi publish lên HiveMQ Cloud (TLS 8883) mỗi 3 giây.    ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import cv2
import base64
import time
import ssl
import os
from dotenv import load_dotenv
import paho.mqtt.client as mqtt

# Load cấu hình từ file .env
load_dotenv()

# ====================================================================
# HẰNG SỐ CẤU HÌNH – Tự động tải từ file .env
# ====================================================================
MQTT_BROKER   = os.getenv("MQTT_BROKER")
MQTT_PORT     = int(os.getenv("MQTT_PORT", 8883))
MQTT_USERNAME = os.getenv("MQTT_USERNAME")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD")

# Topic MQTT dùng chung giữa camera và server AI    
TOPIC_CAMERA  = "haui/smartdoor/camera"

# Thông số truyền ảnh
JPEG_QUALITY  = 70    # Chất lượng nén JPEG (%) – giảm băng thông
SEND_INTERVAL = 3     # Chu kỳ gửi ảnh (giây)
CAMERA_INDEX  = 0     # Index webcam (0 = camera mặc định)


def create_mqtt_client() -> mqtt.Client:
    """
    Khởi tạo MQTT client với kết nối TLS bảo mật tới HiveMQ Cloud.

    Nguyên lý:
    - HiveMQ Cloud bắt buộc kết nối qua TLS (port 8883).
    - Ta tạo SSL context với giao thức TLSv1.2 và xác thực chứng chỉ
      CA mặc định của hệ điều hành để mã hoá toàn bộ đường truyền.
    - Callback on_connect xác nhận trạng thái kết nối thành công.
    """
    # Tạo client với giao thức MQTTv5
    client = mqtt.Client(
        client_id="ESP32CAM_Mock",
        protocol=mqtt.MQTTv5
    )

    # Thiết lập xác thực tài khoản
    client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)

    # Cấu hình TLS – sử dụng chứng chỉ CA mặc định của hệ điều hành
    # tls_version=TLSv1.2 đảm bảo tương thích và bảo mật
    client.tls_set(
        ca_certs=None,
        certfile=None,
        keyfile=None,
        cert_reqs=ssl.CERT_REQUIRED,
        tls_version=ssl.PROTOCOL_TLSv1_2
    )

    # Callback khi kết nối thành công
    def on_connect(client, userdata, flags, rc, properties=None):
        if rc == 0:
            print("[MQTT] ✅ Kết nối HiveMQ Cloud thành công!")
        else:
            print(f"[MQTT] ❌ Kết nối thất bại, mã lỗi: {rc}")

    client.on_connect = on_connect
    client.connect(MQTT_BROKER, MQTT_PORT)
    client.loop_start()  # Chạy vòng lặp mạng trên thread riêng

    return client


def capture_and_publish(client: mqtt.Client):
    """
    Vòng lặp chính: Đọc webcam → Nén JPEG → Mã hoá Base64 → Publish MQTT.

    Nguyên lý hoạt động:
    1. cv2.VideoCapture mở webcam với index đã cấu hình.
    2. Mỗi chu kỳ SEND_INTERVAL giây, đọc 1 khung hình (frame).
    3. cv2.imencode nén frame thành JPEG với chất lượng JPEG_QUALITY.
       - Giảm kích thước dữ liệu đáng kể so với ảnh RAW, mô phỏng
         giới hạn băng thông thực tế của ESP32-CAM.
    4. base64.b64encode chuyển dữ liệu nhị phân JPEG sang chuỗi ASCII
       để truyền an toàn qua payload MQTT (tránh lỗi ký tự đặc biệt).
    5. Publish chuỗi Base64 lên topic camera để server AI subscribe.
    """
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("[CAM] ❌ Không thể mở webcam! Kiểm tra kết nối camera.")
        return

    print(f"[CAM] 📷 Webcam đã sẵn sàng (index={CAMERA_INDEX})")
    print(f"[CAM] 🔄 Bắt đầu truyền ảnh mỗi {SEND_INTERVAL}s...")
    print("-" * 55)

    frame_count = 0

    # Biến theo dõi thời gian gửi MQTT – khởi tạo = 0 để gửi ngay frame đầu tiên
    last_publish_time = 0.0

    try:
        while True:
            # Bước 1: Đọc khung hình từ webcam liên tục (không bị chặn)
            ret, frame = cap.read()
            if not ret:
                print("[CAM] ⚠️  Không đọc được frame, thử lại...")
                continue

            # Bước 2: Hiển thị luồng video trực tiếp lên màn hình
            # Gọi liên tục mỗi vòng lặp → video mượt mà, không bị giật
            cv2.imshow("Camera Gia lap", frame)

            # Bước 3: Bắt sự kiện bàn phím – thoát nhanh bằng phím 'q'
            # cv2.waitKey(1) chờ 1ms để cập nhật cửa sổ GUI và đọc phím.
            # Phép AND 0xFF lấy 8 bit thấp để tương thích mọi hệ điều hành.
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[CAM] 🛑 Nhấn phím 'q' – Thoát chương trình.")
                break

            # Bước 4: Kiểm tra chu kỳ gửi MQTT (non-blocking)
            # Thay vì dùng time.sleep(3) gây chặn toàn bộ vòng lặp,
            # ta so sánh thời gian hiện tại với lần gửi cuối cùng.
            # Chỉ khi đủ SEND_INTERVAL giây mới thực hiện nén + gửi ảnh.
            # Nhờ vậy cv2.imshow() vẫn được gọi liên tục → video mượt.
            current_time = time.time()
            if current_time - last_publish_time >= SEND_INTERVAL:

                # Bước 4a: Nén ảnh sang JPEG với chất lượng đã cấu hình
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
                ret_encode, jpeg_buffer = cv2.imencode('.jpg', frame, encode_params)

                if not ret_encode:
                    print("[CAM] ⚠️  Nén JPEG thất bại, bỏ qua frame.")
                    continue

                # Bước 4b: Mã hoá JPEG binary → Base64 string (UTF-8)
                b64_string = base64.b64encode(jpeg_buffer.tobytes()).decode('utf-8')

                # Bước 4c: Publish chuỗi Base64 lên topic MQTT
                result = client.publish(TOPIC_CAMERA, b64_string)
                frame_count += 1

                # Cập nhật mốc thời gian gửi cuối cùng
                last_publish_time = current_time

                # In thông tin giám sát
                size_kb = len(b64_string) / 1024
                print(
                    f"[CAM] 📤 Frame #{frame_count:04d} | "
                    f"Kích thước: {size_kb:.1f} KB | "
                    f"MQTT rc: {result.rc}"
                )

    except KeyboardInterrupt:
        # Xử lý khi người dùng nhấn Ctrl+C để thoát
        print("\n[CAM] 🛑 Nhận tín hiệu ngắt (Ctrl+C).")

    finally:
        # === KHỐI GIẢI PHÓNG TÀI NGUYÊN AN TOÀN ===
        # Đảm bảo camera, cửa sổ GUI và kết nối mạng luôn được đóng
        # đúng cách, tránh rò rỉ tài nguyên ngay cả khi có lỗi.
        cap.release()
        print("[CAM] 📷 Đã giải phóng webcam.")

        # Đóng tất cả cửa sổ OpenCV (imshow) để giải phóng giao diện
        cv2.destroyAllWindows()
        print("[CAM] 🖥️  Đã đóng cửa sổ hiển thị.")

        client.loop_stop()
        client.disconnect()
        print("[MQTT] 🔌 Đã ngắt kết nối MQTT.")
        print(f"[CAM] 📊 Tổng số frame đã gửi: {frame_count}")


# ====================================================================
# ĐIỂM VÀO CHƯƠNG TRÌNH
# ====================================================================
if __name__ == "__main__":
    print("=" * 55)
    print("  MOCK ESP32-CAM – Hệ thống Khóa cửa Thông minh")
    print("=" * 55)

    mqtt_client = create_mqtt_client()

    # Chờ 2 giây để kết nối MQTT ổn định trước khi truyền ảnh
    time.sleep(2)

    capture_and_publish(mqtt_client)
