from PIL import Image
import os

def resize_images(input_folder, output_folder, size=(2048, 1024)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path)
            img_resized = img.resize(size, Image.LANCZOS)
            img_resized.save(os.path.join(output_folder, filename))
            print(f"Resized and saved {filename}")

# 폴더 경로 설정
input_folder = './input_image'  # 원본 이미지 폴더 경로
output_folder = './output_image'  # 리사이즈된 이미지가 저장될 폴더 경로

# 이미지 리사이즈 및 저장 실행
resize_images(input_folder, output_folder)
