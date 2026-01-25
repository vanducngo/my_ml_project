from processor.services import RFMEngine
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse
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

    customer_id = str(customer_id).strip()
    try:
        # Thử convert sang float trước để bắt trường hợp '12345.0'
        float_val = float(customer_id)
        # Kiểm tra xem nó có phải số nguyên không (vd: 123.0 là OK, 123.5 là Fail)
        if not float_val.is_integer():
            return {"status": "error", "message": "Customer ID must be an integer (e.g., 12345)"}
        
        # Nếu OK, chuẩn hóa về dạng số nguyên string (bỏ .0)
        customer_id = str(int(float_val))
    except ValueError:
        # Nếu không thể convert sang float -> chắc chắn chứa ký tự chữ
        return {"status": "error", "message": f"Invalid Customer ID format: '{customer_id}'. Must contains digits only."}

    # --- Code xử lý logic transaction ở đây ---
    print(f"Đang xử lý giao dịch cho khách hàng: {customer_id}")
    
    engine = RFMEngine()
    result = engine.predict_customer(customer_id)

    print(f"new_transaction -> Result: {result}")

    return JsonResponse(result)


# API 2: retrain_all
@api_view(['POST'])
def retrain_all(request):
    print("Bắt đầu train lại toàn bộ hệ thống...")
    
    # --- Giả lập quá trình train ---
    # time.sleep(2) 
    engine = RFMEngine()
    result = engine.train()
    
    return JsonResponse(result)


# API 3: retrain_classifier
@api_view(['POST'])
def relabel_all(request):
    print("relabel_all...")
    
    engine = RFMEngine()
    result = engine.predict()
    
    return JsonResponse(result)