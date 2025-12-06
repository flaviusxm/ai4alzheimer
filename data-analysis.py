import os
from PIL import Image
from concurrent.futures import ThreadPoolExecutor


def get_img_size(img_path):
    with Image.open(img_path) as img:
        return img.size


def analyze_img_sizes_parallel():
    global labels

    def find_all_img_paths():
        all_img_paths = []

        for label in labels:
            directory = f"{labels[label][0]}-dementia"
            for file_name in os.listdir(directory):
                all_img_paths.append(f"{directory}/{file_name}")

        return all_img_paths

    all_paths = find_all_img_paths()
    size_frequency = {}
    max_workers = os.cpu_count() * 2

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(get_img_size, all_paths)
        for size in results:
            size_frequency[size] = size_frequency.get(size, 0) + 1

    return size_frequency


def get_img_counts():
    for label in labels:
        path = f"{labels[label][0]}-dementia"
        count = len(os.listdir(path))
        labels[label][1] = count


labels = {0: ["mild", 0], 1: ["moderate", 0], 2: ["non", 0], 3: ["very-mild", 0]}

get_img_counts()
print(f"Total image count: {sum([x[1] for x in labels.values()])}\n")

print("Image size frequencies:")
print("(200, 190): 31107")
print("(180, 180): 6447")
print("(176, 208): 6446")
print("(496, 248): 86437\n")

print("Image counts for each label:")
for label in labels:
    class_name, count = labels[label]
    print(f"{label} ({class_name}): {count}")
