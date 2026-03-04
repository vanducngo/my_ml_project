from processor.services_with_validation import RFMEngine
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse
import time
import json
import requests
import threading
from django.views.decorators.csrf import csrf_exempt
from django.core.cache import cache


CALLBACK_URL = "http://61.28.226.98:9000/api/label_change/"

def send_webhook(data):
    """Hàm chạy ngầm để gửi dữ liệu đi mà không block request chính"""
    try:
        if not data:
            return
            
        print(f"--- Đang gửi Webhook tới {CALLBACK_URL} ({len(data)} records) ---")
        response = requests.post(
            CALLBACK_URL, 
            json=data, 
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        print(f"--- Webhook Response: {response.status_code} ---")
    except Exception as e:
        print(f"--- Lỗi gửi Webhook: {e} ---")

# API 1: new_transaction (Nhận customer_id)
@api_view(['POST'])
def new_transaction(request):
    cache.clear()
    # Lấy customer_id từ dữ liệu gửi lên (Body JSON)
    customer_id = request.data.get('customer_id')

    if not customer_id:
        return Response(
            {"error": "Thiếu tham số customer_id"}, 
            status=status.HTTP_400_BAD_REQUEST
        )

    try:
        customer_id = customer_id.strip()
        # --- Code xử lý logic transaction ở đây ---
        print(f"Đang xử lý giao dịch cho khách hàng: {customer_id}")
        
        engine = RFMEngine()
        result = engine.predict_customer(customer_id)

        if result.get('status') == 'success':
            print(f"new_transaction success => Start send backdata: {len(result['data'])}")
            threading.Thread(target=send_webhook, args=(result['data'],)).start()
        
        # print(f"new_transaction -> Result: {result}")
        return JsonResponse(result)
    except Exception as e:
        print(f'Exception: {e}')
        return Response(
                {"status": "error", "message": f"Error - '{e}'"},
                status=status.HTTP_400_BAD_REQUEST
            )     


# API 2: retrain_all
@api_view(['POST'])
def retrain_all(request):
    cache.clear()
    # Lấy customer_id từ dữ liệu gửi lên (Body JSON)
    retrain_history_id = request.data.get('retrain_history_id')

    if not retrain_history_id:
        return Response(
            {"error": "Thiếu tham số customer_id"}, 
            status=status.HTTP_400_BAD_REQUEST
        )
    
    print("Bắt đầu train lại toàn bộ hệ thống...")
    
    # --- Giả lập quá trình train ---
    # time.sleep(2) 
    engine = RFMEngine()
    result = engine.train(retrain_history_id)
    
    if result.get('status') == 'success':
        print(f"retrain_all success => Start send backdata: {len(result['data'])}")
        threading.Thread(target=send_webhook, args=(result['data'],)).start()

    return JsonResponse(result)


# API 3: retrain_classifier
@api_view(['POST'])
def relabel_all(request):
    cache.clear()
    print("relabel_all...")
    
    engine = RFMEngine()
    result = engine.predict()
    if result.get('status') == 'success':
        print(f"relabel_all success => Start send backdata: {len(result['data'])}")
        threading.Thread(target=send_webhook, args=(result['data'],)).start()
    return JsonResponse(result)