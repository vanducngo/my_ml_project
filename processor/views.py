from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
import time

# API 1: new_transaction (Nhận customer_id)
@api_view(['POST'])
def new_transaction(request):
    # Lấy customer_id từ dữ liệu gửi lên (Body JSON)
    customer_id = request.data.get('customer_id')

    if not customer_id:
        return Response(
            {"error": "Thiếu tham số customer_id"}, 
            status=status.HTTP_400_BAD_REQUEST
        )

    # --- Code xử lý logic transaction ở đây ---
    print(f"Đang xử lý giao dịch cho khách hàng: {customer_id}")
    
    return Response({
        "message": "Giao dịch đã được ghi nhận",
        "customer_id": customer_id,
        "status": "success"
    }, status=status.HTTP_201_CREATED)


# API 2: retrain_all
@api_view(['POST'])
def retrain_all(request):
    print("Bắt đầu train lại toàn bộ hệ thống...")
    
    # --- Giả lập quá trình train ---
    # time.sleep(2) 
    
    return Response({
        "message": "Đã hoàn thành Retrain All",
        "accuracy": 0.98  # Ví dụ trả về độ chính xác giả
    }, status=status.HTTP_200_OK)


# API 3: retrain_classifier
@api_view(['POST'])
def retrain_classifier(request):
    print("Chỉ train lại Classifier...")
    
    # --- Logic train classifier ở đây ---

    return Response({
        "message": "Đã hoàn thành Retrain Classifier",
        "model_version": "v2.5"
    }, status=status.HTTP_200_OK)