import hashlib

# Hàm 1: Băm nát mật khẩu để cất vào két sắt
def get_password_hash(password: str):
    return hashlib.sha256(password.encode()).hexdigest()

# Hàm 2: Kiểm tra mật khẩu lúc Đăng nhập xem có khớp không
def verify_password(plain_password, hashed_password):
    return get_password_hash(plain_password) == hashed_password

# Hàm 3: Tạo token (Tạm thời để trống, anh em mình sẽ dùng sau)
def create_access_token(data: dict):
    pass