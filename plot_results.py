import json
import os

import matplotlib.pyplot as plt
import pandas as pd

# --- CẤU HÌNH ---
OUTPUT_DIR = "outputs"
IMG_DIR = "outputs"  # Lưu ảnh vào cùng thư mục outputs
# Đảm bảo thư mục tồn tại
os.makedirs(IMG_DIR, exist_ok=True)


# --- HÀM VẼ BIỂU ĐỒ 1: ROBUSTNESS CURVE (GWCC) ---
def plot_robustness_curve():
    print("Dang ve bieu do Robustness Curve (GWCC)...")

    # Giả sử bạn có 3 file log cho 3 kịch bản (bạn cần đổi tên file cho đúng thực tế)
    # Nếu chưa có file thật, script sẽ tạo dữ liệu giả lập (dummy data) để test
    scenarios = [
        {
            "name": "Random Failure",
            "file": "attack_log_random.json",
            "color": "green",
            "marker": "o",
        },
        {
            "name": "Targeted (Degree)",
            "file": "attack_log_degree.json",
            "color": "blue",
            "marker": "^",
        },
        {
            "name": "Targeted (Collective Influence)",
            "file": "attack_log_ci.json",
            "color": "red",
            "marker": "x",
        },
    ]

    plt.figure(figsize=(10, 6))

    for sc in scenarios:
        filepath = os.path.join(OUTPUT_DIR, sc["file"])

        # Kiểm tra file có tồn tại không. Nếu không -> Tạo dữ liệu giả để Demo
        if not os.path.exists(filepath):
            print(f"Warning: Khong tim thay {sc['file']}. Dang dung du lieu gia lap...")
            # Dữ liệu giả: Random giảm chậm, Targeted giảm nhanh
            x = [0.0, 0.05, 0.1, 0.15, 0.2]
            if "Random" in sc["name"]:
                y = [1.0, 0.98, 0.95, 0.90, 0.85]
            elif "Degree" in sc["name"]:
                y = [1.0, 0.80, 0.60, 0.40, 0.20]
            else:  # CI
                y = [1.0, 0.70, 0.40, 0.15, 0.05]
        else:
            # Đọc dữ liệu thật từ JSON
            with open(filepath) as f:
                data = json.load(f)
            # Giả sử cấu trúc JSON là list các bước: [{"fraction_removed": 0.1, "gwcc": 0.9}, ...]
            df = pd.DataFrame(data)
            x = df["fraction_removed"]
            y = df["gwcc"]

        plt.plot(
            x,
            y,
            label=sc["name"],
            color=sc["color"],
            marker=sc["marker"],
            linestyle="-",
            linewidth=2,
            markersize=6,
        )

    plt.title("Network Robustness Analysis: GWCC Degradation", fontsize=14)
    plt.xlabel("Fraction of Nodes Removed", fontsize=12)
    plt.ylabel("Giant Weakly Connected Component (GWCC) Size", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize=11)

    # Lưu file đúng tên yêu cầu trong LaTeX
    save_path = os.path.join(IMG_DIR, "robustness_curve_gwcc.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"-> Da luu: {save_path}")
    plt.close()


# --- HÀM VẼ BIỂU ĐỒ 2: EFFICIENCY DROP (ASPL) ---
def plot_efficiency_drop():
    print("Dang ve bieu do Efficiency Drop (ASPL)...")

    # Chỉ vẽ cho trường hợp Targeted Attack (vì nó ảnh hưởng ASPL rõ nhất)
    filepath = os.path.join(OUTPUT_DIR, "attack_log_degree.json")  # File ví dụ

    plt.figure(figsize=(10, 6))

    if not os.path.exists(filepath):
        print("Warning: Khong tim thay log file. Dang dung du lieu gia lap...")
        x = [0.0, 0.05, 0.1, 0.15, 0.2]
        y = [3.5, 3.8, 4.5, 5.2, 6.0]  # ASPL tăng dần
    else:
        with open(filepath) as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        x = df["fraction_removed"]
        y = df["aspl"]

    plt.plot(
        x,
        y,
        color="purple",
        marker="s",
        linestyle="-",
        linewidth=2,
        label="Targeted Attack (Degree)",
    )
    plt.fill_between(x, y, min(y), color="purple", alpha=0.1)  # Tô màu vùng dưới

    plt.title("Impact of Attacks on Network Efficiency (ASPL)", fontsize=14)
    plt.xlabel("Fraction of Nodes Removed", fontsize=12)
    plt.ylabel("Average Shortest Path Length (Hops)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()

    # Lưu file đúng tên yêu cầu trong LaTeX
    save_path = os.path.join(IMG_DIR, "efficiency_drop_aspl.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"-> Da luu: {save_path}")
    plt.close()


# --- MAIN ---
if __name__ == "__main__":
    plot_robustness_curve()
    plot_efficiency_drop()
    print("\nHOAN TAT! Hay kiem tra thu muc 'outputs/' de lay anh.")
