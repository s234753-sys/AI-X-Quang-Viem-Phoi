# Dùng môi trường Python 3.9
FROM python:3.9

# Thiết lập thư mục làm việc
WORKDIR /code

# Copy file requirements.txt vào và cài đặt thư viện
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Tạo user để chạy server (luật bảo mật của Hugging Face)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy toàn bộ code vào server
COPY --chown=user . $HOME/app

# Khởi động Uvicorn ở cổng 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]