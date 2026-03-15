import io
import base64
import numpy as np
import tensorflow as tf
import cv2
import csv
from fastapi.responses import StreamingResponse
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from database import engine, Base, get_db, User, History
from security import get_password_hash, verify_password
from fastapi import Form, Depends, HTTPException
from sqlalchemy.orm import Session

# Câu thần chú đúc két sắt SQLite (Tự động tạo file .db)
Base.metadata.create_all(bind=engine)
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# 1. Tải mô hình
model = tf.keras.models.load_model("vip_pro_pneumonia_DENSENET.keras")
# Tự động tìm lớp Convolution cuối cùng (Mắt thần)
last_conv_layer_name = None
for layer in reversed(model.layers):
    try:
        if len(layer.output.shape) == 4:
            last_conv_layer_name = layer.name
            break
    except:
        continue

# 2. Hàm thuật toán Grad-CAM (Đã Bọc thép chống lỗi)
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    try:
        if last_conv_layer_name is None:
            return None
            
        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
        )
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            
            if isinstance(preds, list):
                preds = preds[0]
            if isinstance(last_conv_layer_output, list):
                last_conv_layer_output = last_conv_layer_output[0]
                
            preds = tf.convert_to_tensor(preds)
            
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, last_conv_layer_output)
        
        # --- CHỐNG LỖI NONE SẬP NGUỒN Ở ĐÂY ---
        if grads is None:
            return None
            
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()
    except Exception as e:
        print("Lỗi tạo Mắt thần:", e)
        return None

# 3. Hàm chèn bản đồ nhiệt lên ảnh gốc (Đã sửa lỗi tàng hình)
def create_overlay(image, heatmap):
    # 1. Resize Mắt thần bằng đúng kích thước ảnh gốc
    image_np = np.array(image.convert("RGB"))
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    heatmap_resized = cv2.resize(heatmap, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_CUBIC)

    # 2. Ma thuật Cánh Bướm: Ép sáng 2 lá phổi, dìm vai và bụng
    rows, cols = heatmap_resized.shape
    x = np.linspace(-1, 1, cols)
    y = np.linspace(-1, 1, rows)
    X, Y = np.meshgrid(x, y)
    
    left_lung = np.exp(-((X + 0.45)**2 / 0.15 + (Y - 0.1)**2 / 0.35))
    right_lung = np.exp(-((X - 0.45)**2 / 0.15 + (Y - 0.1)**2 / 0.35))
    spatial_mask = np.maximum(left_lung, right_lung)
    
    heatmap_masked = heatmap_resized * spatial_mask
    
    # 3. Bơm máu màu đỏ nhiệt vào ảnh
    if np.max(heatmap_masked) > 0:
        heatmap_masked = heatmap_masked / np.max(heatmap_masked)
        
    heatmap_masked = np.where(heatmap_masked > 0.35, heatmap_masked, 0)
    heatmap_uint8 = np.uint8(255 * heatmap_masked)
    jet = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    
    # 4. Ép 2 lớp ảnh đè lên nhau (Alpha Blending)
    alpha = heatmap_masked[..., np.newaxis] 
    superimposed_img = (jet * alpha * 0.6) + (image_bgr * (1 - alpha * 0.6))
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    
    # Xuất thành chuỗi nén gửi về Web
    _, buffer = cv2.imencode('.jpg', superimposed_img)
    return base64.b64encode(buffer).decode('utf-8')

# 4. API Phân tích
@app.post("/predict/")
async def predict(
    file: UploadFile = File(...), 
    gender: str = Form(...), 
    patient_name: str = Form(...),
    username: str = Form(...), 
    threshold: float = Form(75.0),
    db: Session = Depends(get_db)
):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    
    img_resized = img.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    
   # --- ĐIỂM SỬA SỐ 1: Trả lại Kính lúp xịn của DenseNet ---
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.densenet.preprocess_input(img_array)
    # --------------------------------------------------------------
    
    # AI bắt đầu dự đoán
    preds = model.predict(img_array)
    
    print("============= ĐIỂM THẬT CỦA AI V2 =============")
    print("Điểm thô (Raw preds):", preds)
    print("===============================================")
    
    if isinstance(preds, list):
        preds = preds[0]
    preds_array = preds[0]
    
   # THUẬT TOÁN BẮT BỆNH THEO CHỈ THỊ TỪ THANH TRƯỢT
    if len(preds_array) >= 2:
        pneumonia_score = float(preds_array[1])
    else:
        pneumonia_score = float(preds_array[0])
        # Nếu model 1 cột, điểm càng nhỏ càng là Viêm phổi
        if pneumonia_score <= 0.5:
            pneumonia_score = 1.0 - pneumonia_score

    # Đổi cái số % bác sĩ cài (vd: 80) thành số thập phân (0.8) để so với AI
    threshold_ratio = threshold / 100.0
    
    if pneumonia_score >= threshold_ratio:
        label = "Viêm phổi"
        confidence = pneumonia_score * 100
        pred_index = 1 if len(preds_array) >= 2 else 0
    else:
        label = "Bình thường"
        # Báo bình thường nhưng ghi kèm "nghi ngờ" nếu điểm AI khá cao nhưng chưa qua ngưỡng
        if pneumonia_score > 0.5: 
            label = "Bình thường (Cần theo dõi)"
            
        confidence = (1 - pneumonia_score) * 100 if pneumonia_score < 0.5 else pneumonia_score * 100
        pred_index = 0
            
    heatmap_b64 = ""
    # --- ĐIỂM SỬA SỐ 2: Chỉ bật Mắt thần khi có bệnh Viêm phổi ---
    if label == "Viêm phổi":
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index)
        if heatmap is not None:
            heatmap_b64 = create_overlay(img, heatmap)

    confidence_str = f"{confidence:.2f}%"

    # --- LƯU KẾT QUẢ VÀO DATABASE ---
    try:
        new_record = History(
            patient_name=patient_name,
            result=label, 
            confidence=confidence_str, 
            doctor_email=username # Điểm ăn tiền: Lấy gói 'username' nhét vào cột 'doctor_email'
        )
        db.add(new_record)
        db.commit()
    except Exception as e:
        print(f"Lỗi khi lưu Database: {e}")
    # --------------------------------
    return {
        "prediction": label,
        "confidence": confidence_str,
        "heatmap_image": f"data:image/jpeg;base64,{heatmap_b64}" if heatmap_b64 else ""
    }

# --- API ĐĂNG KÝ TÀI KHOẢN ---
@app.post("/register/")
def register_user(
    fullname: str = Form(...),
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    # 1. Kiểm tra xem tên đăng nhập này có ai xài chưa
    db_user = db.query(User).filter(User.username == username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Tên đăng nhập đã tồn tại! Vui lòng chọn tên khác.")
    
    # 2. Băm nát mật khẩu ra để bảo mật
    hashed_password = get_password_hash(password)
    
    # 3. Tạo hồ sơ Bác sĩ mới và nhét vào két sắt Database
    new_user = User(fullname=fullname, username=username, hashed_password=hashed_password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    return {"message": "Đăng ký tài khoản thành công!", "username": new_user.username}

# --- API ĐĂNG NHẬP TÀI KHOẢN ---
@app.post("/login/")
def login_user(
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    # 1. Tìm xem tài khoản này có trong két sắt không?
    db_user = db.query(User).filter(User.username == username).first()
    if not db_user:
        raise HTTPException(status_code=400, detail="Tên đăng nhập không tồn tại!")
    
    # 2. Lấy mật khẩu nhập vào so sánh với mật khẩu băm trong két
    if not verify_password(password, db_user.hashed_password):
        raise HTTPException(status_code=400, detail="Mật khẩu không chính xác!")
    
    # 3. Mở cửa thành công!
    return {"message": "Đăng nhập thành công!", "username": db_user.username, "fullname": db_user.fullname}

# --- API LẤY LỊCH SỬ BỆNH ÁN (ĐÃ NÂNG CẤP) ---
@app.get("/history/{username}")
def get_history(username: str, db: Session = Depends(get_db)):
    # Lọc hồ sơ dựa trên cột doctor_email mới (sắp xếp từ mới nhất đến cũ nhất)
    records = db.query(History).filter(History.doctor_email == username).order_by(History.id.desc()).all()
    return records
# --- API TRÍCH XUẤT EXCEL (CSV) ---
@app.get("/export-excel/{username}")
def export_excel(username: str, db: Session = Depends(get_db)):
    # 1. Lục két sắt lấy toàn bộ bệnh án của Bác sĩ này
    records = db.query(History).filter(History.doctor_email == username).order_by(History.id.desc()).all()
    
    # 2. Tạo một file bảng tính trong bộ nhớ RAM
    stream = io.StringIO()
    stream.write('\ufeff')  # Bùa chú (BOM) để Excel đọc tiếng Việt không bị lỗi font chữ
    writer = csv.writer(stream)
    
    # 3. Viết dòng tiêu đề (Header)
    writer.writerow(["Mã Bệnh Án", "Tên Bệnh Nhân", "Kết Quả Trí Tuệ Nhân Tạo", "Độ Tin Cậy"]) 
    
    # 4. Đổ dữ liệu từ Database vào từng dòng
    for r in records:
        writer.writerow([f"BA-{r.id:04d}", r.patient_name, r.result, r.confidence])
        
    # 5. Gói lại thành file .csv và gửi thẳng về máy tính của Bác sĩ
    response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
    response.headers["Content-Disposition"] = f"attachment; filename=Bao_Cao_Y_Te_{username}.csv"
    return response

# --- API XÓA BỆNH ÁN (DELETE) ---
@app.delete("/history/{record_id}")
def delete_history(record_id: int, db: Session = Depends(get_db)):
    # 1. Tìm bệnh án theo ID
    record = db.query(History).filter(History.id == record_id).first()
    
    if not record:
        raise HTTPException(status_code=404, detail="Không tìm thấy bệnh án!")
    
    # 2. Xé bỏ và lưu lại két sắt
    db.delete(record)
    db.commit()
    
    return {"message": "Đã xóa bệnh án thành công!"}