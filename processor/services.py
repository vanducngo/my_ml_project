import os
import joblib
import datetime
import json
import requests
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import xgboost as xgb
from django.conf import settings
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Thư viện kết nối DB
from sqlalchemy import create_engine, text

class RFMEngine:
    def __init__(self):
        # Đường dẫn tuyệt đối tới các thư mục
        self.DATA_DIR = os.path.join(settings.BASE_DIR, 'data')
        self.ARTIFACT_DIR = os.path.join(settings.BASE_DIR, 'ml_artifacts')
        
        # Đảm bảo thư mục tồn tại
        os.makedirs(self.DATA_DIR, exist_ok=True)
        os.makedirs(self.ARTIFACT_DIR, exist_ok=True)

        # Định nghĩa các file cần lưu/load
        self.files = {
            'model': os.path.join(self.ARTIFACT_DIR, 'xgboost_model.pkl'),
            'scaler': os.path.join(self.ARTIFACT_DIR, 'scaler.pkl'),
            # Không cần encoder vì train trực tiếp trên Cluster ID (số)
            # 'encoder': os.path.join(self.ARTIFACT_DIR, 'encoder.pkl'), 
            'config': os.path.join(self.ARTIFACT_DIR, 'config.json'),
        }

        # Cấu hình DB Odoo (Docker container name hoặc localhost)
        self.DATA_API_URL = "http://61.28.226.98:9000/api/get_transactions/"

    def _load_data_from_api(self):
        """
        Hàm thay thế load_db: Tải toàn bộ dữ liệu từ API có phân trang.
        Cơ chế: Loop liên tục tăng offset cho đến khi returned_records = 0.
        """
        
        all_records = []
        limit = 5000  # Tăng limit lên để giảm số lần request (Network overhead)
        offset = 0
        page = 1
        
        print(f"--- Đang tải dữ liệu từ API: {self.DATA_API_URL} Limit:{limit}-Offset:{offset}---")
        try:
            while True:
                # Gọi API
                params = {'limit': limit, 'offset': offset}
                response = requests.get(self.DATA_API_URL, params=params, timeout=30)
                
                if response.status_code != 200:
                    raise Exception(f"API Error {response.status_code}: {response.text}")
                
                data = response.json()
                metadata = data.get('metadata', {})
                records = data.get('records', [])
                
                count = len(records)
                if count == 0:
                    break  # Hết dữ liệu
                
                all_records.extend(records)
                print(f"Page {page}: Loaded {count} records (Offset: {offset})")
                
                offset += count
                page += 1
                
                # Safety break (đề phòng vòng lặp vô tận nếu API lỗi)
                if metadata.get('returned_records', 0) == 0:
                    break

            print(f"--- Tải hoàn tất: {len(all_records)} dòng dữ liệu ---")
            
            # Convert list of dicts to DataFrame
            df = pd.DataFrame(all_records)
            return df
            
        except Exception as e:
            print(f"LỖI TẢI DỮ LIỆU API: {e}")
            raise e
        
    def _preprocessing(self, df):
        print("Tiền xử lý dữ liệu - Start")
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

        targetDf = df.loc[~df.duplicated()]             
        targetDf = targetDf[targetDf['Quantity'] > 0]   
        targetDf = targetDf[targetDf['Price'] > 0]      
        targetDf = targetDf[targetDf['Customer ID'].notna()]    

        print("Tiền xử lý dữ liệu - End")
        return targetDf

    def _calculate_rfm(self, targetDf):
        """Hàm tính toán RFM từ dữ liệu giao dịch thô"""
        latest_date = targetDf['InvoiceDate'].max() + datetime.timedelta(days=1)

        recency_df = targetDf.groupby('Customer ID').agg({
            'InvoiceDate': lambda x: (latest_date - x.max()).days
        }).rename(columns={'InvoiceDate': 'Recency'})

        frequency_df = targetDf.groupby('Customer ID').agg({
            'Invoice': 'nunique'
        }).rename(columns={'Invoice': 'Frequency'})

        targetDf['TotalPrice'] = targetDf['Quantity'] * targetDf['Price']
        monetary_df = targetDf.groupby('Customer ID').agg({
            'TotalPrice': 'sum'
        }).rename(columns={'TotalPrice': 'Monetary'})

        rfm_df = recency_df.join(frequency_df).join(monetary_df)
        rfm_df.reset_index(inplace=True)
        
        return rfm_df
    
    def remove_outliers_iqr(self, df, columns):
        # Trả về dataframe mới để tránh warning SettingWithCopy
        df_clean = df.copy()
        initial_count = df_clean.shape[0]

        for col in columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]

        print(f"Số điểm ngoại lai đã bị loại bỏ: {initial_count - df_clean.shape[0]}")
        return df_clean

    def _format_for_webhook(self, df_result):
        """Helper format JSON trả về"""
        results = []
        for _, row in df_result.iterrows():
            results.append({
                "customer_id": str(row['Customer ID']),
                "label": row['Segment'],
                "recency_score": int(row['Recency']),       
                "frequency_score": int(row['Frequency']),   
                "monetary_score": float(round(row['Monetary'], 2))
            })
        return results

    def train(self, csv_filename='online_retail_II.csv', use_api=False):
        """
        Quy trình huấn luyện Model từ đầu.
        Có thể chọn nguồn dữ liệu từ CSV hoặc Database.
        """
        print("--- Bắt đầu quy trình Training ---")
        
        # 1. Load Dữ Liệu (DB hoặc CSV)
        if use_api:
            try:
                print("LoadData Prom API")
                df = self._load_data_from_api()
            except Exception as e:
                return {"status": "error", "message": str(e)}
        else:
            csv_path = os.path.join(self.DATA_DIR, csv_filename)
            if not os.path.exists(csv_path):
                return {"error": f"File {csv_filename} không tồn tại."}
            df = pd.read_csv(csv_path)

        # 2. Preprocessing & RFM
        df = self._preprocessing(df)
        rfm_df = self._calculate_rfm(df)

        # 3. Remove Outliers
        rfm_df = self.remove_outliers_iqr(rfm_df, ['Recency', 'Frequency', 'Monetary'])

        # 4. Data Transformation (Box-Cox & Log)
        rfm_processed = pd.DataFrame()
        rfm_processed['Customer ID'] = rfm_df['Customer ID']

        # Fix lỗi Box-Cox với số <= 0
        if (rfm_df['Recency'] <= 0).any():
            rfm_df['Recency'] = rfm_df['Recency'].apply(lambda x: 1 if x <= 0 else x)
            
        if (rfm_df['Frequency'] <= 0).any():
            rfm_df['Frequency'] = rfm_df['Frequency'].apply(lambda x: 1 if x <= 0 else x)

        # Lưu lại lambda để dùng cho predict
        rfm_processed['Recency'], lmbda_r = stats.boxcox(rfm_df['Recency'])
        rfm_processed['Frequency'], lmbda_f = stats.boxcox(rfm_df['Frequency'])
        rfm_processed['Monetary'] = np.cbrt(rfm_df['Monetary']).values

        # 5. Scaling
        scaler = StandardScaler()
        rfm_scaled_vals = scaler.fit_transform(rfm_processed[['Recency', 'Frequency', 'Monetary']])
        df_rfm_scaled = pd.DataFrame(rfm_scaled_vals, columns=['Recency', 'Frequency', 'Monetary'])
        
        # 6. K-Means Clustering
        optimal_k = 5
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(df_rfm_scaled[['Recency', 'Frequency', 'Monetary']])
        
        rfm_full = rfm_df.copy()
        rfm_full['Cluster'] = clusters # Cluster ID: 0, 1, 2, 3, 4 (Ngẫu nhiên)

        # 7. Gán nhãn tự động (Auto-Mapping)
        print("--- Bắt đầu gán nhãn tự động (Auto-Mapping) ---")
        
        # Tính trung bình R, F, M cho từng cụm
        summary = rfm_full.groupby('Cluster').agg({
            'Recency': 'mean', 
            'Frequency': 'mean', 
            'Monetary': 'mean'
        })
        
        available_clusters = list(summary.index)
        label_map = {}

        # Rule 1: Tiền nhiều nhất -> VIP
        # Logic: Cụm có Monetary Cao Nhất
        vip_id = summary.loc[available_clusters]['Monetary'].idxmax()
        label_map[int(vip_id)] = 'Khách hàng VIP'
        available_clusters.remove(vip_id)

        # Rule 2: Recency cao nhất (lâu không mua) -> Rời bỏ
        # Logic: Cụm có Recency Cao Nhất (Lâu chưa mua)
        lost_id = summary.loc[available_clusters]['Recency'].idxmax()
        label_map[int(lost_id)] = 'Nguy cơ rời bỏ'
        available_clusters.remove(lost_id)

        # --- Rule 3: Khách hàng Mới ---
        # Logic: Min(Recency) + Min(Monetary)
        # Ta tính rank (thứ hạng) cho R và M trong các cụm còn lại.
        # Cụm nào có tổng hạng thấp nhất -> Mới
        current_subset = summary.loc[available_clusters]
        # Rank tăng dần (nhỏ nhất là 1)
        r_rank = current_subset['Recency'].rank(ascending=True)
        m_rank = current_subset['Monetary'].rank(ascending=True)
        # Tổng điểm (Score càng thấp càng khớp tiêu chí Mới: R thấp, M thấp)
        new_score = r_rank + m_rank
        new_id = new_score.idxmin()
        label_map[int(new_id)] = 'Khách hàng Mới'
        available_clusters.remove(new_id)

        # --- Rule 4: Khách hàng Trung thành ---
        # Logic: Min(Recency) trong các cụm còn lại
        # (Không phải Mới, không phải VIP, nhưng mua gần đây -> Trung thành)
        loyal_id = summary.loc[available_clusters]['Recency'].idxmin()
        label_map[int(loyal_id)] = 'Khách hàng Trung thành'
        available_clusters.remove(loyal_id)

        # Rule 5: Còn lại -> Tiềm năng
        potential_id = available_clusters[0]
        label_map[int(potential_id)] = 'Khách hàng Tiềm năng'

        # print("Mapping Logic được tạo ra:", label_map)
        
        # Áp dụng mapping vào DataFrame để visual (nếu cần)
        rfm_full['Segment'] = rfm_full['Cluster'].map(label_map)

        # ==========================================================
        # 8. Train XGBoost (Train trên Cluster ID)
        # ==========================================================
        print("--- Huấn luyện XGBoost trên Cluster IDs ---")
        
        X = df_rfm_scaled[['Recency', 'Frequency', 'Monetary']]
        y = rfm_full['Cluster'] # Train target là số (0,1,2,3,4)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        print(classification_report(y_test, y_pred))

        # ==========================================================
        # 9. Lưu Artifacts & Config
        # ==========================================================
        joblib.dump(model, self.files['model'])
        joblib.dump(scaler, self.files['scaler'])
        # Không cần lưu encoder nữa vì ta train bằng số int trực tiếp
        
        config = {
            'lambda_recency': lmbda_r,
            'lambda_frequency': lmbda_f,
            'label_map': label_map # Lưu mapping để dùng khi predict
        }
        
        with open(self.files['config'], 'w') as f:
            json.dump(config, f)

        
        # =========================================================
        # TỐI ƯU HÓA: DỰ ĐOÁN LUÔN TRÊN TOÀN BỘ DATA ĐÃ TẢI
        # (Không gọi lại API, dùng biến rfm_df_all)
        # =========================================================
        print("--- Đang thực hiện gán nhãn lại cho TOÀN BỘ dữ liệu ---")
        
        rfm_predict = rfm_df.copy()
        
        # Transform & Predict
        rfm_predict['Recency'] = rfm_predict['Recency'].apply(lambda x: 1 if x <= 0 else x)
        rfm_predict['Frequency'] = rfm_predict['Frequency'].apply(lambda x: 1 if x <= 0 else x)
        
        rfm_predict['Recency'] = stats.boxcox(rfm_predict['Recency'], lmbda=lmbda_r)
        rfm_predict['Frequency'] = stats.boxcox(rfm_predict['Frequency'], lmbda=lmbda_f)
        rfm_predict['Monetary'] = np.cbrt(rfm_predict['Monetary'])
        
        X_all_new = scaler.transform(rfm_predict[['Recency', 'Frequency', 'Monetary']])
        all_clusters = model.predict(X_all_new)
        
        rfm_df['Segment'] = pd.Series(all_clusters).map(label_map)
        
        # Format kết quả
        webhook_data = self._format_for_webhook(rfm_df)

        return {
            "status": "success", 
            "message": "Retrain & Relabel thành công.",
            "data": webhook_data
        }

    def predict_customer(self, customer_id, csv_filename='online_retail_II.csv'):
        """
        Dự đoán phân khúc cho MỘT khách hàng cụ thể.
        Trả về cả Cluster ID (Số) và Segment Name (Chữ).
        """
        print(f"--- Bắt đầu dự đoán cho Customer ID: {customer_id} ---")

        # 1. Kiểm tra Model & Config
        if not os.path.exists(self.files['model']):
            return {"status": "error", "message": "Model chưa được train. Hãy chạy /train/ trước."}

        # 2. Đọc file CSV gốc (Lấy lịch sử giao dịch)
        # csv_path = os.path.join(self.DATA_DIR, csv_filename)
        # if not os.path.exists(csv_path):
        #     return {"status": "error", "message": f"File {csv_filename} không tồn tại."}
        
        # df = pd.read_csv(csv_path)
        

        api_url = f"{self.DATA_API_URL}?customer_id={customer_id}"
        try:
            response = requests.get(api_url, timeout=10)
            print(f"API: {api_url}")
            print(f"Response: {response}")
            if response.status_code != 200:
                 return {"status": "error", "message": f"API Error: {response.status_code}"}
            
            data = response.json()
            records = data.get('records', [])
            
            if not records:
                return {"status": "error", "message": f"Không tìm thấy giao dịch nào của KH {customer_id}"}
                
            df = pd.DataFrame(records)
            
        except Exception as e:
            return {"status": "error", "message": f"Lỗi gọi API: {str(e)}"}

        # 3. Preprocessing nhanh
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        df = df[df['Quantity'] > 0]
        df = df[df['Price'] > 0]
        df = df.dropna(subset=['Customer ID'])

        # 4. Xác định ngày mốc toàn cục
        global_latest_date = datetime.datetime.now() + datetime.timedelta(days=1)

        # 5. Lọc dữ liệu Customer
        str_customer_id = str(int(float(customer_id))) # Chuẩn hóa ID
        df['Customer ID'] = df['Customer ID'].astype(float).astype(int).astype(str)
        
        customer_df = df[df['Customer ID'] == str_customer_id]

        if customer_df.empty:
            return {"status": "error", "message": f"Không tìm thấy Customer ID {customer_id}"}

        # 6. Tính RFM
        recency = (global_latest_date - customer_df['InvoiceDate'].max()).days
        frequency = customer_df['Invoice'].nunique()
        monetary = (customer_df['Quantity'] * customer_df['Price']).sum()

        # Xử lý an toàn cho Box-Cox
        recency = 1 if recency <= 0 else recency
        frequency = 1 if frequency <= 0 else frequency
        monetary = 0.001 if monetary <= 0 else monetary

        # 7. Load Model & Config
        model = joblib.load(self.files['model'])
        scaler = joblib.load(self.files['scaler'])
        with open(self.files['config'], 'r') as f:
            config = json.load(f)

        # Load Mapping từ config (Convert key từ string sang int)
        label_map = {int(k): v for k, v in config['label_map'].items()}

        # 8. Transform & Scale
        try:
            r_trans = stats.boxcox([recency], lmbda=config['lambda_recency'])
            f_trans = stats.boxcox([frequency], lmbda=config['lambda_frequency'])
            m_trans = np.cbrt([monetary])
            
            # Tạo array 2D cho scaler
            rfm_processed = np.array([[r_trans[0], f_trans[0], m_trans[0]]])
            X_new = scaler.transform(rfm_processed)
            
        except Exception as e:
            return {"status": "error", "message": f"Lỗi tính toán: {str(e)}"}

        # 9. Predict & Map Result
        pred_cluster_id = int(model.predict(X_new)[0]) # Ra số (ví dụ: 4)
        segment_name = label_map.get(pred_cluster_id, "Không xác định")

        # 10. Trả về kết quả (đã ép kiểu native python để tránh lỗi JSON)
        result_item = {
            "customer_id": str(customer_id),
            "label": segment_name, # Biến segment_name lấy từ logic predict cũ
            "recency_score": int(recency),
            "frequency_score": int(frequency),
            "monetary_score": float(round(monetary, 2))
        }

        result_item = {
            "customer_id": str(customer_id),
            "label": segment_name,
            "recency_score": int(recency),
            "frequency_score": int(frequency),
            "monetary_score": float(round(monetary, 2))
        }

        return {
            "status": "success",
            "data": [result_item]
        }
    
    def predict(self, csv_filename='online_retail_II.csv', use_db=False):
        """
        Quy trình dự đoán phân khúc cho TOÀN BỘ khách hàng (Batch Prediction).
        Input: File CSV giao dịch hoặc lấy từ DB.
        Output: File CSV kết quả chứa Customer ID, RFM, Cluster ID và Segment.
        """
        print("--- Bắt đầu quy trình Batch Prediction ---")
        
        # 1. Kiểm tra Model & Config
        if not os.path.exists(self.files['model']) or not os.path.exists(self.files['config']):
            return {"status": "error", "message": "Model chưa được train. Hãy chạy /train/ trước."}

        # 2. Load Dữ Liệu
        try:
            df = self._load_data_from_api()
        except Exception as e:
            return {"status": "error", "message": str(e)}
        
        # 3. Load Artifacts
        model = joblib.load(self.files['model'])
        scaler = joblib.load(self.files['scaler'])
        with open(self.files['config'], 'r') as f:
            config = json.load(f)

        # 4. Tiền xử lý & Tính RFM (Giống hệt quy trình Train)
        # Bắt buộc phải chạy _preprocessing để lọc data rác
        df = self._preprocessing(df)
        rfm_df = self._calculate_rfm(df)
        
        if rfm_df.empty:
            return {"status": "error", "message": "Không đủ dữ liệu hợp lệ để tính RFM"}
            
        # 5. Xử lý dữ liệu an toàn cho toán học (Box-Cox yêu cầu dương)
        rfm_process = rfm_df.copy()
        
        # Xử lý Recency <= 0 (nếu có)
        rfm_process['Recency'] = rfm_process['Recency'].apply(lambda x: 1 if x <= 0 else x)
        # Xử lý Frequency <= 0 (ít khi xảy ra nhưng cần đề phòng)
        rfm_process['Frequency'] = rfm_process['Frequency'].apply(lambda x: 1 if x <= 0 else x)
        # Xử lý Monetary <= 0 (Căn bậc 3 chịu được số âm, nhưng log thì không. Ở đây ta dùng cbrt nên an toàn)
        
        # 6. Áp dụng Transformation (Dùng tham số từ Config)
        try:
            # Lưu ý: stats.boxcox(x, lmbda=...) trả về array đã transform
            rfm_process['Recency'] = stats.boxcox(rfm_process['Recency'], lmbda=config['lambda_recency'])
            rfm_process['Frequency'] = stats.boxcox(rfm_process['Frequency'], lmbda=config['lambda_frequency'])
            rfm_process['Monetary'] = np.cbrt(rfm_process['Monetary'])
        except Exception as e:
            return {"status": "error", "message": f"Lỗi biến đổi dữ liệu: {str(e)}"}

        # 7. Áp dụng Scaling (Dùng Scaler đã train)
        X_new = scaler.transform(rfm_process[['Recency', 'Frequency', 'Monetary']])

        # 8. Dự đoán Cluster ID
        # Kết quả trả về là mảng các số nguyên [0, 4, 2, 1...]
        cluster_ids = model.predict(X_new)

        # 9. Mapping sang Segment Name
        # Config lưu keys dưới dạng string "0", "1"... cần convert về int
        label_map = {int(k): v for k, v in config['label_map'].items()}
        
        rfm_df['Cluster_ID'] = cluster_ids
        rfm_df['Segment'] = rfm_df['Cluster_ID'].map(label_map)

        # 10. Lưu kết quả
        # output_filename = f"prediction_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        # output_path = os.path.join(self.DATA_DIR, output_filename)
        # rfm_df.to_csv(output_path, index=False)

        # print(f"--- Dự đoán xong. Đã lưu tại: {output_filename} ---")

        # 11. Trả về Preview (5 dòng đầu)
        # Convert sang dict và xử lý các kiểu dữ liệu numpy để tránh lỗi JSON
        webhook_data = self._format_for_webhook(rfm_df)
        
        return {
            "status": "success",
            "data": webhook_data
        }
    
    def _format_for_webhook(self, df_result):
        """
        Helper chuyển DataFrame thành List Dict:
        [
            {
                "customer_id": "13085",
                "label": "Mới",
                "recency_score": 5,
                "frequency_score": 2,
                "monetary_score": 3
            }, ...
        ]
        """
        results = []
        for _, row in df_result.iterrows():
            results.append({
                "customer_id": str(int(float(row['Customer ID']))) if str(row['Customer ID']).replace('.','').isdigit() else str(row['Customer ID']),
                "label": row['Segment'],
                "recency_score": int(row['Recency']),       # Giá trị ngày thực tế
                "frequency_score": int(row['Frequency']),   # Số lần mua thực tế
                "monetary_score": float(round(row['Monetary'], 2)) # Số tiền thực tế
            })
        return results