import pandas as pd 
file_path = 'C:/sin/대 4 신영/23년 2학기/딥러닝/경진대회/데이터/zzz/Zzzs_train_multi.parquet'
df = pd.read_parquet(file_path)
df.info()
print(df.head())

# series_id 값으로 그룹화하여 저장
grouped = df.groupby('series_id')

# 그룹별로 반복하여 CSV 파일로 저장
for series_id, group_data in grouped:
    csv_file_name = f'zzzs_train_multi_series_{series_id}.csv'  # 파일 이름 생성
    group_data.to_csv("zzzs_train_multi_series/"+csv_file_name, index=True)  # CSV 파일로 저장

print(10)