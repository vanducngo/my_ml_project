import os
import joblib
import datetime
import json

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import xgboost as xgb
from django.conf import settings
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

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
            'encoder': os.path.join(self.ARTIFACT_DIR, 'encoder.pkl'),
            'config': os.path.join(self.ARTIFACT_DIR, 'config.json'),
        }

    def _preprocessing(self, df):
        print("Tiền xử lý dữ liệu - Start")
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

        targetDf = df.loc[~df.duplicated()]             # Xóa duplicated data
        targetDf = targetDf[targetDf['Quantity'] > 0]   # Xoá hết những đơn mà có số lượng mua item nhỏ hơn hoạc bằng không
        targetDf = targetDf[targetDf['Price'] > 0]      # Xoá hết những đơn mà có số giá item nhỏ hơn hoạc bằng không
        targetDf = targetDf[targetDf['Customer ID'].notna()]    # Xoá những đơn hàng không có Customer ID

        print("Tiền xử lý dữ liệu - End")
        return targetDf

    def _calculate_rfm(self, targetDf):
        """Hàm tính toán RFM từ dữ liệu giao dịch thô"""
        
        # # Tạo cột TotalPrice
        # df['TotalPrice'] = df['Quantity'] * df['Price']
        
        # # Chọn ngày mốc (ngày cuối cùng trong data + 1)
        # latest_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
        
        # # GroupBy tính RFM
        # rfm = df.groupby('Customer ID').agg({
        #     'InvoiceDate': lambda x: (latest_date - x.max()).days,
        #     'Invoice': 'nunique',
        #     'TotalPrice': 'sum'
        # }).rename(columns={
        #     'InvoiceDate': 'Recency',
        #     'Invoice': 'Frequency',
        #     'TotalPrice': 'Monetary'
        # })
        
        # # Xử lý dữ liệu âm hoặc bằng 0 (Bắt buộc cho Box-Cox)
        # rfm = rfm[rfm['Monetary'] > 0]
        # rfm['Recency'] = rfm['Recency'].apply(lambda x: 1 if x <= 0 else x)
        
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
        # Lưu số lượng dòng ban đầu
        initial_count = df.shape[0]

        for col in columns:
            # Tính toán Q1 (quartile 25) và Q3 (quartile 75)
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)

            # Tính toán IQR (khoảng giữa Q1 và Q3)
            IQR = Q3 - Q1

            # Xác định giá trị ngoại lai (dưới Q1 - 1.5*IQR hoặc trên Q3 + 1.5*IQR)
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Loại bỏ các dòng có giá trị ngoại lai
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

        # Lưu số lượng dòng sau khi loại bỏ
        final_count = df.shape[0]

        # Tính toán số điểm ngoại lai đã bị loại bỏ
        outliers_removed = initial_count - final_count

        print(f"Số điểm ngoại lai đã bị loại bỏ: {outliers_removed}")

        return df

    def train(self, csv_filename='online_retail_II.csv'):
        """Quy trình huấn luyện Model từ đầu"""
        print("--- Bắt đầu quy trình Training ---")
        
        # 1. Đọc CSV
        csv_path = os.path.join(self.DATA_DIR, csv_filename)
        if not os.path.exists(csv_path):
            return {"error": f"File {csv_filename} không tồn tại trong thư mục data/"}
            
        df = pd.read_csv(csv_path)

        df = self._preprocessing(df)
        
        # 2. Tính RFM
        rfm_df = self._calculate_rfm(df)

        # rfm_df = self.remove_outliers_iqr(df)
        rfm_df = self.remove_outliers_iqr(df, ['Recency', 'Frequency', 'Monetary'])

        # Transform the data while keeping 'Customer ID'
        rfm_processed = pd.DataFrame()

        # Copy 'Customer ID' to the new dataframe
        rfm_processed['Customer ID'] = rfm_df['Customer ID']

        # Apply transformations to Recency, Frequency, and Monetary
        rfm_processed['Recency'], lmbda_r = stats.boxcox(rfm_df['Recency'])[0]
        rfm_processed['Frequency'], lmbda_f = stats.boxcox(rfm_df['Frequency'])[0]
        rfm_processed['Monetary'] = pd.Series(np.cbrt(rfm_df['Monetary'])).values

        # 4. Chuẩn hóa (Scaling)
        scaler = StandardScaler()
        # Chuẩn hóa các trường cần thiết
        df_rfm_scaled_values = scaler.fit_transform(rfm_processed[['Recency', 'Frequency', 'Monetary']])

        # Chuyển dữ liệu chuẩn hóa thành DataFrame, giữ nguyên tên cột
        df_rfm_scaled = pd.DataFrame(df_rfm_scaled_values, columns=['Recency', 'Frequency', 'Monetary'])

        # Gắn lại cột Customer ID
        df_rfm_scaled['Customer ID'] = rfm_processed['Customer ID'].values

        # Đưa cột Customer ID lên đầu
        df_rfm_scaled = df_rfm_scaled[['Customer ID', 'Recency', 'Frequency', 'Monetary']]
        
        # 5. Phân cụm (K-Means) để tạo nhãn (Labeling)
        # Giả định K=5 như bài toán đã chốt
        optimal_k = 5
        # Giữ lại DataFrame gốc chứa Customer ID
        df_rfm_scaled_df = df_rfm_scaled.copy()

        # Tạo bản sao chỉ chứa các cột cần thiết cho phân cụm
        df_for_clustering = df_rfm_scaled[['Recency', 'Frequency', 'Monetary']]

        # Thực hiện phân cụm
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        kmeans.fit(df_for_clustering)

        # Thêm nhãn Cluster vào DataFrame gốc
        df_rfm_scaled_df['Cluster'] = kmeans.labels_

        # Merge hai DataFrame dựa trên Customer ID
        rfm_df_new = pd.merge(rfm_df[['Customer ID', 'Recency', 'Frequency', 'Monetary']],
                            df_rfm_scaled_df[['Customer ID', 'Cluster']],
                            on='Customer ID',
                            how='left')
        
        cluster_summary = rfm_df_new.groupby('Cluster').agg(
            Recency_mean=('Recency', 'mean'),
            Frequency_mean=('Frequency', 'mean'),
            Monetary_mean=('Monetary', 'mean'),
            Customer_count=('Customer ID', 'count')
        ).reset_index()


        # 2. Đổi tên các cột để khớp với bảng trong bài báo
        cluster_summary.rename(columns={
            'Cluster': 'Cluster ID',
            'Recency_mean': 'R (Thấp)',
            'Frequency_mean': 'F (Cao)',
            'Monetary_mean': 'M (Cao)',
            'Customer_count': 'Số lượng KH'
        }, inplace=True)

        # 3. Gán nhãn nghiệp vụ cho từng cụm dựa trên phân tích
        # (Bạn có thể điều chỉnh các nhãn này cho phù hợp với kết quả thực tế của mình)
        segment_labels_map = {
            0: 'Khách hàng VIP',
            1: 'Khách hàng Nguy cơ rời bỏ', # Dựa trên R cao, F/M thấp
            2: 'Khách hàng Trung thành',
            3: 'Khách hàng Tiềm năng',
            4: 'Khách hàng Mới'
        }
        cluster_summary['Nhãn đề xuất'] = cluster_summary['Cluster ID'].map(segment_labels_map)


        # 4. Sắp xếp lại các cột theo đúng thứ tự như trong bài báo
        final_table = cluster_summary[[
            'Cluster ID',
            'R (Thấp)',
            'F (Cao)',
            'M (Cao)',
            'Số lượng KH',
            'Nhãn đề xuất'
        ]].set_index('Cluster ID')
        
        
        rfm_df_final = rfm_df_new
        print("--- Bắt đầu gán nhãn nghiệp vụ cho các phân khúc ---")

        # Tạo từ điển ánh xạ từ số Cluster sang Tên Phân khúc
        # Dựa trên phân tích đã thống nhất ở trên.
        label_map = {
            0: 'Khách hàng VIP',
            4: 'Khách hàng Mới',
            2: 'Khách hàng Trung thành',
            1: 'Nguy cơ rời bỏ',
            3: 'Khách hàng tiềm năng'
        }

        # Sử dụng hàm .map() để tạo cột 'Segment' mới
        rfm_df_final['Segment'] = rfm_df_final['Cluster'].map(label_map)

        print("\n--- Gán nhãn hoàn tất! ---")


        # 7. Huấn luyện XGBoost (Supervised Learning)
        # Mục đích: Để sau này predict nhanh hơn, không cần chạy lại K-Means
        print("--- Bắt đầu chuẩn bị, mã hóa và chia dữ liệu ---")

        # 1. Xác định Features (X)
        X = rfm_df_final[['Recency', 'Frequency', 'Monetary']]

        # 2. Xác định và Mã hóa Target (y)
        y_original = rfm_df_final['Cluster']

        # Khởi tạo và fit LabelEncoder để học các nhãn chuỗi
        le = LabelEncoder()
        y_encoded = le.fit_transform(y_original)

        # In ra để xem ánh xạ đã tạo
        print("Ánh xạ từ nhãn chuỗi sang số:")
        for i, class_name in enumerate(le.classes_):
            print(f"'{class_name}' -> {i}")

        # 3. Chia dữ liệu đã được mã hóa
        # Sử dụng y_encoded để chia
        X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        print("\n--- Chia dữ liệu hoàn tất! ---")
        print(f"Kích thước tập huấn luyện (X_train): {X_train.shape}")
        print(f"Kích thước tập kiểm tra (X_test):   {X_test.shape}")

        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train_encoded)
        y_test_encoded = le.transform(y_test_encoded)
        
        model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')

        # Huấn luyện mô hình
        model.fit(X_train, y_train_encoded)

        # Dự đoán trên tập test
        y_pred_encoded = model.predict(X_test)

        # Chuyển đổi ngược nhãn để báo cáo
        y_pred_original = le.inverse_transform(y_pred_encoded)
        y_test_original = le.inverse_transform(y_test_encoded)

        # In báo cáo đánh giá
        print(f"Báo cáo phân loại cho mô hình XGBClassifier:")
        print(classification_report(y_test_original, y_pred_original))

        # 8. LƯU ARTIFACTS (Quan trọng nhất)
        joblib.dump(model, self.files['model'])
        joblib.dump(scaler, self.files['scaler'])
        joblib.dump(le, self.files['encoder'])
        
        config = {
            'lambda_recency': lmbda_r,
            'lambda_frequency': lmbda_f
        }
        with open(self.files['config'], 'w') as f:
            json.dump(config, f)

        return {
            "status": "success", 
            "message": "Training completed & Model saved.",
        }

    def predict(self, csv_filename='new_transactions.csv'):
        """Quy trình dự đoán cho dữ liệu mới"""
        print("--- Bắt đầu quy trình Prediction ---")
        
        # 1. Kiểm tra file Model
        if not os.path.exists(self.files['model']):
            return {"error": "Model chưa được train. Hãy chạy /train/ trước."}

        # 2. Đọc dữ liệu mới
        csv_path = os.path.join(self.DATA_DIR, csv_filename)
        if not os.path.exists(csv_path):
            return {"error": f"File {csv_filename} không tồn tại."}
            
        df_new = pd.read_csv(csv_path)
        
        # 3. Load Artifacts
        model = joblib.load(self.files['model'])
        scaler = joblib.load(self.files['scaler'])
        encoder = joblib.load(self.files['encoder'])
        with open(self.files['config'], 'r') as f:
            config = json.load(f)

        # 4. Tính RFM cho data mới
        rfm_df = self._calculate_rfm(df_new)
        if rfm_df.empty:
            return {"error": "Không đủ dữ liệu để tính RFM"}
            
        # 5. Áp dụng Transformation (Giống hệt lúc Train)
        # Quan trọng: Dùng tham số lambda từ config, KHÔNG tính lại lambda mới
        rfm_process = rfm_df.copy()
        
        rfm_process['Recency'] = stats.boxcox(rfm_process['Recency'], lmbda=config['lambda_recency'])
        rfm_process['Frequency'] = stats.boxcox(rfm_process['Frequency'], lmbda=config['lambda_frequency'])
        rfm_process['Monetary'] = np.cbrt(rfm_process['Monetary'])

        # 6. Áp dụng Scaling
        X_new = scaler.transform(rfm_process[['Recency', 'Frequency', 'Monetary']])

        # 7. Dự đoán
        preds_encoded = model.predict(X_new)
        preds_label = encoder.inverse_transform(preds_encoded)

        # 8. Tổng hợp kết quả
        rfm_df['Segment'] = preds_label
        
        # Lưu kết quả ra file CSV để người dùng xem
        output_path = os.path.join(self.DATA_DIR, 'prediction_results.csv')
        rfm_df.to_csv(output_path)

        # Trả về JSON 5 dòng đầu để preview
        return {
            "status": "success",
            "output_file": output_path,
            "preview": rfm_df.head().to_dict(orient='index')
        }