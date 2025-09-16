import requests
import os
import random

API_KEY = "tH9T85WPUnDPOm9TPhRiekm4iNUyfGhGGDTr4UewuQigworetVoU9SsX"  
N_IMAGES = 13
SAVE_DIR = r"D:\jigsaw\dataset2"

def get_max_number_in_filenames(folder):
    max_num = 0
    if not os.path.exists(folder):
        return max_num
        
    for filename in os.listdir(folder):
        try:
            num = int(os.path.splitext(filename)[0])
            if num > max_num:
                max_num = num
        except ValueError:
            continue
    return max_num

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

start_num = get_max_number_in_filenames(SAVE_DIR) + 1

url = "https://api.pexels.com/v1/curated"
headers = {"Authorization": API_KEY}
params = {
    "per_page": min(N_IMAGES, 80),
    "page": random.randint(1, 100)
}

try:
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    data = response.json()

    photos = data.get("photos", [])
    if not photos:
        print("Не удалось получить изображения. Проверьте API-ключ или соединение.")
        exit(1)

    selected_photos = random.sample(photos, min(N_IMAGES, len(photos)))
    for i, photo in enumerate(selected_photos):
        img_url = photo["src"]["original"]
        img_response = requests.get(img_url)
        img_response.raise_for_status()

        file_extension = img_url.split(".")[-1]
        file_path = os.path.join(SAVE_DIR, f"{start_num + i}.{file_extension}")

        with open(file_path, "wb") as f:
            f.write(img_response.content)
        print(f"Скачано: {file_path}")

    print(f"Успешно скачано {len(selected_photos)} изображений в папку {SAVE_DIR}")

except requests.exceptions.RequestException as e:
    print(f"Ошибка при скачивании: {e}")
except Exception as e:
    print(f"Произошла ошибка: {e}")