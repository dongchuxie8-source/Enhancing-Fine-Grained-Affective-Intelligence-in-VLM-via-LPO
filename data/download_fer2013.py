"""
FER2013数据集下载和预处理脚本
"""
import pandas as pd
import numpy as np
from pathlib import Path
import os
import sys


def download_fer2013():
    """下载FER2013数据集"""
    print("=" * 50)
    print("下载FER2013数据集")
    print("=" * 50)

    # 检查是否已安装kaggle
    try:
        import kaggle
    except ImportError:
        print("错误：未安装kaggle库")
        print("请运行: pip install kaggle")
        print("然后配置Kaggle API: https://github.com/Kaggle/kaggle-api")
        sys.exit(1)

    # 创建目录
    raw_dir = Path('./data/raw')
    raw_dir.mkdir(parents=True, exist_ok=True)

    # 下载数据集
    print("\n正在从Kaggle下载FER2013...")
    print("注意：需要先配置Kaggle API credentials")
    print("参考：https://github.com/Kaggle/kaggle-api#api-credentials")

    try:
        kaggle.api.dataset_download_files(
            'msambare/fer2013',
      path=str(raw_dir),
          unzip=True
        )
        print("[OK] 下载完成！")
    except Exception as e:
        print(f"[FAIL] 下载失败: {e}")
        print("\n备选方案：")
        print("1. 手动从 https://www.kaggle.com/datasets/msambare/fer2013 下载")
        print("2. 解压到 ./data/raw/ 目录")
        sys.exit(1)


def preprocess_fer2013():
    """预处理FER2013数据"""
    print("\n" + "=" * 50)
    print("预处理FER2013数据")
    print("=" * 50)

    # 读取数据
    csv_path = './data/raw/fer2013.csv'
    if not os.path.exists(csv_path):
        print(f"[FAIL] 错误：找不到文件 {csv_path}")
        print("请先运行下载步骤")
        sys.exit(1)

    print(f"\n正在读取 {csv_path}...")
    df = pd.read_csv(csv_path)

    # 情感标签映射
    emotion_map = {
        0: 'Angry',
        1: 'Disgust',
        2: 'Fear',
        3: 'Happy',
        4: 'Sad',
        5: 'Surprise',
        6: 'Neutral'
    }

    df['emotion_name'] = df['emotion'].map(emotion_map)

    # 添加模拟的强度值（实际项目中应该有真实标注）
    # 这里使用随机值作为示例
    np.random.seed(42)
    df['intensity'] = np.random.uniform(0, 1, len(df))

    # 数据统计
    print(f"\n数据统计：")
    print(f"  总样本数: {len(df)}")
    print(f"  情感分布:")
    for emotion, count in df['emotion_name'].value_counts().items():
        print(f"    {emotion}: {count}")

    # 保存处理后的数据
    processed_dir = Path('./data/processed')
    processed_dir.mkdir(parents=True, exist_ok=True)

    output_path = processed_dir / 'fer2013_processed.csv'
    df.to_csv(output_path, index=False)
    print(f"\n[OK] 处理完成！保存到 {output_path}")

    return df


def main():
    """主函数"""
    print("\n" + "=" * 50)
    print("FER2013 数据准备")
    print("=" * 50)

    # 步骤1：下载数据
    if not os.path.exists('./data/raw/fer2013.csv'):
        download_fer2013()
    else:
        print("\n[OK] FER2013数据已存在，跳过下载")

    # 步骤2：预处理数据
    if not os.path.exists('./data/processed/fer2013_processed.csv'):
        df = preprocess_fer2013()
    else:
        print("\n[OK] 预处理数据已存在，跳过预处理")
        df = pd.read_csv('./data/processed/fer2013_processed.csv')

    print("\n" + "=" * 50)
    print("数据准备完成！")
    print("=" * 50)
    print(f"\n下一步：运行 'python data/build_preference.py' 构建偏好数据集")


if __name__ == "__main__":
    main()
