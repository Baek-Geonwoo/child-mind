import zipfile

# ZIP 파일 경로
zip_file_path = "C:/sin/대 4 신영/23년 2학기/딥러닝/경진대회/데이터/zzz/Zzzs_train_multi.parquet.zip"

# 압축 해제할 위치
extract_path = "C:/sin/대 4 신영/23년 2학기/딥러닝/경진대회/데이터/zzz/"

# ZIP 파일 압축 해제
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print("10")