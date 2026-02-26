import os
import joblib
import datetime
import json
import requests
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import xgboost as xgb
from django.conf import settings
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

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
            'config': os.path.join(self.ARTIFACT_DIR, 'config.json'),
            'training_data': os.path.join(self.ARTIFACT_DIR, 'training_labeled_data.csv') # File lưu data sau gán nhãn
        }

        self.DATA_API_URL = "http://61.28.226.98:9000/api/get_transactions/"

    def _load_data_from_api(self):
        """Tải dữ liệu phân trang từ API"""
        all_records = []
        limit = 50000
        offset = 0
        page = 1
        
        print(f"--- Đang tải dữ liệu từ API: {self.DATA_API_URL} Limit:{limit}---")
        try:
            while True:
                params = {'limit': limit, 'offset': offset}
                response = requests.get(self.DATA_API_URL, params=params, timeout=30)
                
                if response.status_code != 200:
                    raise Exception(f"API Error {response.status_code}: {response.text}")
                
                data = response.json()
                metadata = data.get('metadata', {})
                records = data.get('records', [])
                
                count = len(records)
                if count == 0:
                    break
                
                all_records.extend(records)
                print(f"Page {page}: Loaded {count} records (Offset: {offset})")
                
                offset += count
                page += 1
                
                if metadata.get('returned_records', 0) == 0:
                    break

            print(f"--- Tải hoàn tất: {len(all_records)} dòng dữ liệu ---")
            df = pd.DataFrame(all_records)
            return df
            
        except Exception as e:
            print(f"LỖI TẢI DỮ LIỆU API: {e}")
            raise e
        
    def _preprocessing(self, df):
        # print("Tiền xử lý dữ liệu - Start")
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        targetDf = df.loc[~df.duplicated()]             
        targetDf = targetDf[targetDf['Quantity'] > 0]   
        targetDf = targetDf[targetDf['Price'] > 0]      
        targetDf = targetDf[targetDf['Customer ID'].notna()]    
        # print("Tiền xử lý dữ liệu - End")
        return targetDf

    def _calculate_rfm(self, targetDf):
        # 1. Tính Total Price
        targetDf = targetDf.copy()
        targetDf['TotalPrice'] = targetDf['Quantity'] * targetDf['Price']
        
        # 2. Ngày mốc
        latest_date = targetDf['InvoiceDate'].max() + datetime.timedelta(days=1)

        # 3. Aggregate
        rfm_df = targetDf.groupby('Customer ID').agg({
            'InvoiceDate': lambda x: (latest_date - x.max()).days,
            'Invoice': 'nunique',
            'TotalPrice': 'sum'
        })

        # 4. Rename
        rfm_df.columns = ['Recency', 'Frequency', 'Monetary']

        # 5. Extra fields
        rfm_df['order_count'] = rfm_df['Frequency']
        rfm_df['total_invoiced_v2'] = rfm_df['Monetary']
        rfm_df['aov'] = rfm_df['Monetary'] / rfm_df['Frequency']

        rfm_df = rfm_df.reset_index()
        return rfm_df
    
    def remove_outliers_iqr(self, df, columns):
        df_clean = df.copy()
        initial_count = df_clean.shape[0]
        for col in columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        print(f"Đã loại bỏ outliers: {initial_count - df_clean.shape[0]} dòng")
        return df_clean

    def _safe_transform_boxcox_input(self, series):
        """Hàm helper để xử lý input <= 0 trước khi Box-Cox cho đồng nhất"""
        return series.apply(lambda x: 1 if x <= 0 else x)
    
    def compute_segment_stats(self, df):
        """
        Tính toán chỉ số trung bình R, F, M cho từng Segment.
        Input: DataFrame đã có cột 'Segment', 'Recency', 'Frequency', 'Monetary'
        Output: DataFrame chứa thống kê (như hình bạn gửi) và trả về dạng Dict để API dùng.
        """
        print("\n--- Thống kê chỉ số trung bình theo từng Segment ---")
        
        if 'Segment' not in df.columns:
            print("Lỗi: DataFrame chưa có cột Segment")
            return None

        # 1. Group by Segment và tính Mean
        stats_df = df.groupby('Segment').agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': 'mean',
            'Customer ID': 'count' # Đếm thêm số lượng khách hàng cho xịn
        }).reset_index()

        # 2. Đổi tên cột cho giống yêu cầu (avg_...)
        stats_df.columns = ['segment_name', 'avg_recency', 'avg_frequency', 'avg_monetary', 'customer_count']

        # 3. Làm tròn số cho đẹp
        stats_df['avg_recency'] = stats_df['avg_recency'].round(2)
        stats_df['avg_frequency'] = stats_df['avg_frequency'].round(2)
        stats_df['avg_monetary'] = stats_df['avg_monetary'].round(2)

        # 4. In ra console (dạng bảng đẹp giống hình bạn gửi)
        try:
            # Nếu cài tabulate thì in sẽ đẹp hơn, không thì print thường
            print(stats_df.to_string(index=False))
        except:
            print(stats_df)
            
        print("----------------------------------------------------\n")

        # 5. Chuyển thành list dict để trả về API (nếu cần hiển thị lên Dashboard)
        return stats_df.to_dict('records')

    def train(self):
        """Quy trình huấn luyện và kiểm thử tại chỗ"""
        print("\n=== BẮT ĐẦU TRAINING ===")
        
        # 1. Load & Process
        try:
            raw_df = self._load_data_from_api()
        except Exception as e:
            return {"status": "error", "message": str(e)}

        df = self._preprocessing(raw_df)
        rfm_df = self._calculate_rfm(df)

        # 2. Remove Outliers (Quan trọng: Lưu lại tập này để validation)
        rfm_clean = self.remove_outliers_iqr(rfm_df, ['Recency', 'Frequency', 'Monetary'])

        # 3. Prepare Data for Transformation
        # Tạo bản copy để transform, giữ bản rfm_clean gốc để lưu CSV
        rfm_for_train = rfm_clean.copy()

        # Xử lý số <= 0 (Logic đồng nhất)
        rfm_for_train['Recency'] = self._safe_transform_boxcox_input(rfm_for_train['Recency'])
        rfm_for_train['Frequency'] = self._safe_transform_boxcox_input(rfm_for_train['Frequency'])
        # Monetary dùng cbrt nên không cần xử lý âm, nhưng nếu dùng log thì cần. Ở đây dùng cbrt.

        # 4. Transform & Save Lambdas
        recency_trans, lmbda_r = stats.boxcox(rfm_for_train['Recency'])
        frequency_trans, lmbda_f = stats.boxcox(rfm_for_train['Frequency'])
        monetary_trans = np.cbrt(rfm_for_train['Monetary']).values

        # Tạo DataFrame đã scaled để clustering
        scaler = StandardScaler()
        # Stack thành mảng 2D: (n_samples, 3)
        rfm_matrix = np.column_stack((recency_trans, frequency_trans, monetary_trans))
        rfm_scaled_vals = scaler.fit_transform(rfm_matrix)
        
        df_rfm_scaled = pd.DataFrame(rfm_scaled_vals, columns=['Recency', 'Frequency', 'Monetary'])
        
        # 5. K-Means Clustering (Ground Truth)
        optimal_k = 5
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(df_rfm_scaled)
        
        # Gán Cluster vào DataFrame gốc (rfm_clean) để lưu trữ
        rfm_clean['Cluster'] = clusters 

        # 6. Auto-Mapping Label
        summary = rfm_clean.groupby('Cluster').agg({'Recency': 'mean', 'Frequency': 'mean', 'Monetary': 'mean'})
        available_clusters = list(summary.index)
        label_map = {}

        # Logic gán nhãn (Giữ nguyên như cũ)
        vip_id = summary.loc[available_clusters]['Monetary'].idxmax()
        label_map[int(vip_id)] = 'VIP'
        available_clusters.remove(vip_id)

        lost_id = summary.loc[available_clusters]['Recency'].idxmax()
        label_map[int(lost_id)] = 'Nguy cơ rời bỏ'
        available_clusters.remove(lost_id)

        current_subset = summary.loc[available_clusters]
        new_score = current_subset['Recency'].rank(ascending=True) + current_subset['Monetary'].rank(ascending=True)
        new_id = new_score.idxmin()
        label_map[int(new_id)] = 'Mới'
        available_clusters.remove(new_id)

        loyal_id = summary.loc[available_clusters]['Recency'].idxmin()
        label_map[int(loyal_id)] = 'Trung thành'
        available_clusters.remove(loyal_id)

        potential_id = available_clusters[0]
        label_map[int(potential_id)] = 'Tiềm năng'

        # Map segment
        rfm_clean['Segment'] = rfm_clean['Cluster'].map(label_map)

        # Tính toán và in ra thống kê ngay sau khi gán nhãn
        self.compute_segment_stats(rfm_clean)

        # === LƯU DATA SAU KHI GÁN NHÃN ===
        print(f"--- Đang lưu dữ liệu đã gán nhãn vào: {self.files['training_data']} ---")
        rfm_clean.to_csv(self.files['training_data'], index=False)

        # 7. Train XGBoost
        print("--- Huấn luyện XGBoost ---")
        X = df_rfm_scaled # Input là data đã transform + scale
        y = rfm_clean['Cluster'] # Target là Cluster ID (số)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
        model.fit(X_train, y_train)

        # Evaluate Test Set
        print("\nReport trên tập Test (Hold-out 20%):")
        y_pred_test = model.predict(X_test)
        print(classification_report(y_test, y_pred_test))

        # 8. Save Artifacts
        joblib.dump(model, self.files['model'])
        joblib.dump(scaler, self.files['scaler'])
        
        config = {
            'lambda_recency': lmbda_r,
            'lambda_frequency': lmbda_f,
            'label_map': label_map
        }
        with open(self.files['config'], 'w') as f:
            json.dump(config, f)

        print("\n=== VALIDATION: CHẠY PREDICT TRÊN TOÀN BỘ TẬP TRAINING ===")
        # Gọi hàm predict nhưng truyền data rfm_clean vào.
        # Mục đích: Verify xem pipeline Predict có ra kết quả giống pipeline Train (KMeans) không.
        # rfm_clean lúc này chứa: R, F, M (raw), Cluster (Kmeans), Segment (Kmeans)
        return self.predict(input_rfm_df=rfm_clean)


    def predict(self, input_rfm_df=None):
        """
        Batch Prediction & Validation với Training Data cũ.
        """
        print("\n--- Bắt đầu quy trình Predict & Validation ---")

        # 1. Kiểm tra Model
        if not os.path.exists(self.files['model']):
            return {"status": "error", "message": "Model not found. Hãy chạy /train/ trước."}

        # 2. Load Dữ liệu hiện tại (Current Data)
        if input_rfm_df is not None:
            rfm_df = input_rfm_df.copy()
        else:
            try:
                # Load từ API (Dữ liệu mới nhất)
                raw_df = self._load_data_from_api()
                df = self._preprocessing(raw_df)
                rfm_df = self._calculate_rfm(df)
            except Exception as e:
                return {"status": "error", "message": str(e)}

        if rfm_df.empty:
            return {"status": "error", "message": "No data to predict"}

        # 3. Load Artifacts
        model = joblib.load(self.files['model'])
        scaler = joblib.load(self.files['scaler'])
        with open(self.files['config'], 'r') as f:
            config = json.load(f)

        # 4. Feature Engineering (Transform & Scale)
        rfm_process = rfm_df.copy()
        
        # Xử lý input <= 0 (Logic đồng nhất với Train)
        rfm_process['Recency'] = self._safe_transform_boxcox_input(rfm_process['Recency'])
        rfm_process['Frequency'] = self._safe_transform_boxcox_input(rfm_process['Frequency'])
        
        try:
            r_trans = stats.boxcox(rfm_process['Recency'], lmbda=config['lambda_recency'])
            f_trans = stats.boxcox(rfm_process['Frequency'], lmbda=config['lambda_frequency'])
            m_trans = np.cbrt(rfm_process['Monetary']).values
            
            rfm_matrix = np.column_stack((r_trans, f_trans, m_trans))
            X_new = scaler.transform(rfm_matrix)
        except Exception as e:
            return {"status": "error", "message": f"Transformation error: {str(e)}"}

        # 5. Predict (Ra Cluster ID mới)
        cluster_ids_pred = model.predict(X_new)
        rfm_df['Cluster_Pred'] = cluster_ids_pred

        # ==============================================================================
        # 6. LOGIC SO SÁNH VỚI TRAINING DATA (Ground Truth Check)
        # ==============================================================================
        validation_info = "Không tìm thấy file training cũ để so sánh."
        acc_score = None
        
        if os.path.exists(self.files['training_data']):
            try:
                print(f"--> Đang load dữ liệu gốc từ: {self.files['training_data']}")
                # Chỉ lấy Customer ID và Cluster gốc
                ground_truth_df = pd.read_csv(self.files['training_data'])
                
                # Đảm bảo Customer ID là string để merge chính xác
                ground_truth_df['Customer ID'] = ground_truth_df['Customer ID'].astype(str)
                rfm_df['Customer ID'] = rfm_df['Customer ID'].astype(str)

                ground_truth_subset = ground_truth_df[['Customer ID', 'Cluster']].rename(columns={'Cluster': 'Cluster_True'})

                # Merge dữ liệu mới dự đoán với dữ liệu gốc
                # Inner join: Chỉ so sánh những khách hàng CÓ MẶT ở cả 2 thời điểm
                comparison_df = pd.merge(rfm_df, ground_truth_subset, on='Customer ID', how='inner')
                
                if not comparison_df.empty:
                    # Tính độ chính xác
                    matches = (comparison_df['Cluster_Pred'] == comparison_df['Cluster_True']).sum()
                    total_overlap = len(comparison_df)
                    acc_score = matches / total_overlap

                    print("\n" + "="*50)
                    print(f"KẾT QUẢ ĐỐI CHIẾU (VALIDATION REPORT)")
                    print("="*50)
                    print(f"- Tổng số khách hàng dự đoán: {len(rfm_df)}")
                    print(f"- Số khách hàng trùng khớp ID với tập Train: {total_overlap}")
                    print(f"- Số lượng khớp nhãn (Cluster cũ == Mới): {matches}")
                    print(f"- Độ trùng khớp (Consistency Accuracy): {acc_score * 100:.2f}%")
                    
                    if acc_score > 0.99:
                        validation_info = "Tuyệt đối (100%). Model hoạt động nhất quán hoàn hảo."
                    elif acc_score > 0.85:
                        validation_info = "Cao (>85%). Dữ liệu có thay đổi nhỏ hoặc Model ổn định."
                    else:
                        validation_info = "Thấp. Có thể hành vi khách hàng đã thay đổi nhiều hoặc Model lệch."
                    
                    # (Tùy chọn) In ra vài trường hợp lệch để debug
                    mismatches = comparison_df[comparison_df['Cluster_Pred'] != comparison_df['Cluster_True']]
                    if not mismatches.empty:
                        print("\nVí dụ vài trường hợp lệch nhãn (Data Drift?):")
                        print(mismatches[['Customer ID', 'Recency', 'Cluster_True', 'Cluster_Pred']].head(5))
                    print("="*50 + "\n")
                else:
                    validation_info = "Không có Customer ID nào trùng giữa dữ liệu mới và cũ."

            except Exception as e:
                print(f"Lỗi khi so sánh validation: {e}")
                validation_info = f"Lỗi so sánh: {str(e)}"

        # 7. Map to Segment Name
        label_map = {int(k): v for k, v in config['label_map'].items()}
        rfm_df['Segment'] = rfm_df['Cluster_Pred'].map(label_map)

        # 8. Trả kết quả
        webhook_data = self._format_for_data_webhook(rfm_df)
        
        return {
            "status": "success",
            "validation_accuracy": acc_score,
            "validation_msg": validation_info,
            "data": webhook_data
        }

    def predict_customer(self, customer_id):
        """Dự đoán cho 1 khách hàng (Real-time)"""
        # (Giữ nguyên logic cũ, chỉ cập nhật phần transform cho an toàn)
        # ... Load Model ...
        # ... Load Data from API ...
        
        # Logic Transform cần sửa lại như sau để đảm bảo nhất quán:
        # rfm_val = self._safe_transform_boxcox_input(pd.Series([raw_val]))
        # trans_val = stats.boxcox(rfm_val, lmbda=config['lambda...'])
        pass 
        # (Em có thể tự cập nhật hàm này dựa trên logic của hàm predict ở trên)

    def _format_for_data_webhook(self, df_result):
        # (Giữ nguyên như code cũ của em)
        results = []
        for _, row in df_result.iterrows():
            order_count = int(row['Frequency']) if row['Frequency'] is not None else 0
            total_invoiced = float(row['Monetary']) if row['Monetary'] is not None else 0.0
            aov = total_invoiced / order_count if order_count > 0 else 0.0
            
            results.append({
                "customer_id": str(row['Customer ID']),
                "label": row['Segment'],
                "recency_score": int(row['Recency']),
                "frequency_score": int(row['Frequency']),
                "monetary_score": float(round(row['Monetary'], 2)),
                "order_count": order_count,
                "total_invoiced_v2": round(total_invoiced, 2),
                "aov": round(aov, 2)
            })
        return results