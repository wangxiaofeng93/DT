from tqdm import tqdm
import time

print("普通进度条演示:")
for i in tqdm(range(50), desc="任务A"):
    time.sleep(0.02)

