# 🫁 PneumoScan.AI - Hệ Thống Chẩn Đoán Viêm Phổi Bằng Deep Learning

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-FF6F00.svg)
![Status](https://img.shields.io/badge/Status-Hoàn_thành_100%25-success.svg)

**PneumoScan.AI** là nền tảng y tế số ứng dụng trí tuệ nhân tạo (Mô hình DenseNet) nhằm hỗ trợ các y bác sĩ trong việc tầm soát và phát hiện sớm các dấu hiệu Viêm phổi thông qua ảnh chụp X-Quang ngực. Dự án được thiết kế với giao diện Dashboard hiện đại, mang lại trải nghiệm tối ưu cho không gian làm việc của các cơ sở y tế.

---

## 🌟 TÍNH NĂNG NỔI BẬT (Key Features)

### 1. 🤖 Lõi AI Chẩn Đoán Siêu Tốc
- Sử dụng kiến trúc mạng nơ-ron tích chập sâu **DenseNet** đã được tinh chỉnh (Fine-tuning).
- Tốc độ xử lý siêu nhanh **< 2 giây** cho mỗi ca phân tích.
- Độ chính xác (Accuracy) đạt trên **89.5%** từ tập dữ liệu huấn luyện hơn 5.800 mẫu.

### 2. 👁️ Tầm Nhìn Xuyên Thấu (Mắt Thần Grad-CAM)
- Không chỉ đưa ra kết quả "Có bệnh" hay "Không có bệnh" (Hộp đen), hệ thống còn tích hợp thuật toán **Grad-CAM**.
- Tự động sinh ra **Bản đồ nhiệt (Heatmap)**, bôi đỏ chính xác vùng phổi có tổn thương, giúp bác sĩ dễ dàng đối chiếu và đưa ra quyết định cuối cùng.

### 3. 🎛️ Bảng Điều Khiển "Độ Nhạy" Linh Hoạt
- Bác sĩ nắm toàn quyền quyết định bằng thanh trượt **Ngưỡng tin cậy (Threshold)**.
- Khi cấp bách, có thể chủ động hạ hoặc tăng ngưỡng phần trăm để AI đánh giá khắt khe hơn, loại bỏ triệt để sai số (False Positive / False Negative). Tự động cảnh báo "Cần theo dõi" nếu ở mức nghi ngờ.

### 4. 📊 Quản Lý Két Sắt Y Tế (CRUD & Báo cáo)
- Tích hợp Database an toàn. Mọi lịch sử chẩn đoán đều được lưu lại chi tiết theo từng tài khoản Bác sĩ.
- Hỗ trợ **Trích xuất dữ liệu ra file Excel (.csv)** chỉ với 1 Click, phục vụ hoàn hảo cho việc làm báo cáo, thống kê dịch tễ cuối tháng.

### 5. 🏥 Mạng Lưới Y Tế Liên Kết Thông Minh
- Tích hợp sổ tay tra cứu nhanh các **Bệnh viện chuyên khoa Hô hấp tuyến đầu** trải dài 3 miền Bắc - Trung - Nam (Hà Nội, Huế, Đà Nẵng, TP.HCM, Cần Thơ).
- Có tính năng **Tìm kiếm thông minh (Search box)** lọc dữ liệu tức thời.

### 6. 🌍 Giao Diện Tùy Biến Chuyên Sâu
- Cấu trúc **SPA (Single Page Application)** chuyển tab mượt mà không cần load lại trang.
- Hỗ trợ **Đa ngôn ngữ toàn diện (Anh/Việt)** dịch từ giao diện tĩnh đến dữ liệu động trả về từ Database.
- Tích hợp công tắc đổi màu nền (Dark/Light mode) bảo vệ mắt cho bác sĩ khi trực ca đêm.
- Tích hợp đăng nhập nhanh bằng **Google (SSO)**.

---

## 🛠️ CÔNG NGHỆ SỬ DỤNG (Tech Stack)

* **Trí tuệ nhân tạo (AI/ML):** `TensorFlow`, `Keras`, `OpenCV`, `NumPy`.
* **Backend (Máy chủ):** `Python`, `FastAPI`, `Uvicorn`.
* **Cơ sở dữ liệu:** `SQLite`, `SQLAlchemy`.
* **Bảo mật:** Băm mật khẩu với `Passlib/Bcrypt`, xác thực `JWT Google`.
* **Frontend (Giao diện):** `HTML5`, `CSS3`, `Vanilla JavaScript`, `FontAwesome`.

---

## 🚀 HƯỚNG DẪN CÀI ĐẶT & SỬ DỤNG

> **⚠️ LƯU Ý QUAN TRỌNG VỀ DỮ LIỆU & MÔ HÌNH:** > Do kích thước Model AI và Dataset khá lớn, thầy vui lòng tải tại link Google Drive này: [(https://drive.google.com/drive/folders/1eDL05Qs85qof6_H7-XYYg6m6VdFV4MCg?usp=sharing)]
> Sau khi tải về: 
> 1. Chép file `vip_pro_pneumonia_DENSENET.keras` vào thư mục `backend/`
> 2. Chép thư mục `chest_xray` vào thư mục gốc của dự án.
> 3. Báo cáo bản pdf. 


### Khởi động Máy chủ Backend (Não bộ AI)
👉 Link đồ án: https://s234753-sys.github.io/AI-X-Quang-Viem-Phoi/frontend/index.html