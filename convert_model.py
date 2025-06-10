from tensorflow.keras.models import load_model
import numpy as np
import json

# 載入訓練好的 H5 模型
model = load_model("model.h5")
weights = model.get_weights()

# ✅ 用老師要的格式寫入模型架構
model_arch = [
    {"name": "flatten", "type": "Flatten", "config": {}, "weights": []},
    {"name": "dense_1", "type": "Dense", "config": {"activation": "relu"}, "weights": ["array_0", "array_1"]},
    {"name": "dense_2", "type": "Dense", "config": {"activation": "relu"}, "weights": ["array_2", "array_3"]},
    {"name": "dense_3", "type": "Dense", "config": {"activation": "softmax"}, "weights": ["array_4", "array_5"]}
]

# 儲存模型結構與權重
with open("fashion_mnist.json", "w") as f:
    json.dump(model_arch, f, indent=2)

weight_dict = {f"array_{i}": w for i, w in enumerate(weights)}
np.savez("fashion_mnist.npz", **weight_dict)

print("✅ fashion_mnist.json 與 .npz 轉換完成")
