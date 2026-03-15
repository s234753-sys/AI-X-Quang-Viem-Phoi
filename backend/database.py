from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

# Tạo đường dẫn kết nối đến file medical_records.db (sẽ tự động sinh ra)
SQLALCHEMY_DATABASE_URL = "sqlite:///./medical_records.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# --- KHUÔN MẪU: BẢNG TÀI KHOẢN BÁC SĨ (Đã nâng cấp xịn xò) ---
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    fullname = Column(String, index=True)
    
    # Dùng Email làm định danh chính cho hệ thống Google
    email = Column(String, unique=True, index=True) 
    
    # Cho phép username và mật khẩu được trống (NULL) nếu bác sĩ xài Google
    username = Column(String, unique=True, index=True, nullable=True) 
    hashed_password = Column(String, nullable=True) 
    
    # Đánh dấu nguồn gốc: 'local' (tạo bằng tay) hoặc 'google'
    auth_provider = Column(String, default="local") 

# --- KHUÔN MẪU: BẢNG LỊCH SỬ KHÁM BỆNH (Đã nâng cấp Y khoa) ---
class History(Base):
    __tablename__ = "histories"
    
    id = Column(Integer, primary_key=True, index=True)
    patient_name = Column(String)
    result = Column(String)
    confidence = Column(String)
    
    # Chìa khóa vàng: Dán nhãn hồ sơ này thuộc về Email Bác sĩ nào
    doctor_email = Column(String, index=True) 
    
    # Tính năng ăn tiền: Tự động lưu Ngày/Giờ lúc AI phân tích xong
    created_at = Column(DateTime, default=datetime.datetime.utcnow) 

# Hàm mở kết nối an toàn đến Database
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()