# Sử dụng image Python 3.10 slim để giảm kích thước
FROM python:3.10-slim

# Đặt thư mục làm việc
WORKDIR /app

# Cài đặt các gói hệ thống cần thiết
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Sao chép file requirements.txt
COPY requirements.txt .

# Cài đặt các thư viện Python
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép mã nguồn, thư mục static, dữ liệu và mô hình
COPY app.py .
COPY model.py .
COPY static/ ./static/
COPY data/ ./data/
COPY transformer_model.pth .

# Expose cổng 8000 cho FastAPI
EXPOSE 8000

# Lệnh chạy ứng dụng với uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]