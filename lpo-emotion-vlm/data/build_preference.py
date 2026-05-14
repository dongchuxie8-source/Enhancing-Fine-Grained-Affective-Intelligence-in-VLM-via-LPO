"""
构建有序偏好数据集

根据情感类别和强度值，为每张图片生成排序的描述列表
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import train_test_split


# 情感强度描述模板
INTENSITY_TEMPLATES = {
    'Happy': {
        'high': "This image shows an intensely happy expression with a bright, radiant smile and sparkling eyes.",
        'medium': "This image shows a clearly happy expression with a visible smile.",
     'low': "This image shows a slightly happy expression with a subtle smile.",
        'neutral': "This image shows a nearly neutral expression with minimal emotional cues."
    },
    'Sad': {
        'high': "This image shows an intensely sad expression with furrowed brows and a downturned mouth.",
        'medium': "This image shows a clearly sad expression with a low mood.",
        'low': "This image shows a slightly sad expression with mild melancholy.",
        'neutral': "This image shows a nearly neutral expression with a hint of sadness."
    },
    'Angry': {
        'high': "This image shows an intensely angry expression with tight lips and narrowed eyes.",
        'medium': "This image shows a clearly angry expression with visible tension.",
        'low': "This image shows a slightly angry expression with mild irritation.",
        'neutral': "This image shows a nearly neutral expression with a hint of annoyance."
    },
    'Fear': {
        'high': "This image shows an intensely fearful expression with wide eyes and an open mouth.",
        'medium': "This image shows a clearly fearful expression with visible anxiety.",
        'low': "This image shows a slightly fearful expression with mild unease.",
      'neutral': "This image shows a nearly neutral expression with a hint of concern."
    },
    'Surprise': {
        'high': "This image shows an intensely surprised expression with raised eyebrows and a wide-open mouth.",
        'medium': "This image shows a clearly surprised expression with raised eyebrows.",
      'low': "This image shows a slightly surprised expression with mild astonishment.",
        'neutral': "This image shows a nearly neutral expression with a hint of surprise."
    },
    'Disgust': {
        'high': "This image shows an intensely disgusted expression with a wrinkled nose and curled lip.",
        'medium': "This image shows a clearly disgusted expression with visible aversion.",
     'low': "This image shows a slightly disgusted expression with mild distaste.",
        'neutral': "This image shows a nearly neutral expression with a hint of discomfort."
    },
    'Neutral': {
        'high': "This image shows a completely neutral expression with no emotional cues.",
      'medium': "This image shows a mostly neutral expression with very subtle cues.",
        'low': "This image shows a nearly neutral expression with slight emotional hints.",
        'neutral': "This image shows a neutral expression with ambiguous emotional signals."
    }
}


def get_intensity_level(intensity):
    """将连续强度值映射到离散等级"""
    if intensity > 0.75:
        return 'high'
    elif intensity > 0.5:
        return 'medium'
    elif intensity > 0.25:
        return 'low'
    else:
        return 'neutral'


def build_ranked_texts(emotion, intensity):
    """
    根据情感和强度构建排序描述列表
    
    Args:
        emotion: 情感类别
        intensity: 强度值 [0, 1]
    
    Returns:
        ranked_texts: 按强度从高到低排序的描述列表
    """
    if emotion not in INTENSITY_TEMPLATES:
        emotion = 'Neutral'
    
    templates = INTENSITY_TEMPLATES[emotion]
    
    # 根据真实强度决定排序
    level = get_intensity_level(intensity)
    
    if level == 'high':
        # 最高强度：high > medium > low > neutral
        ranked_texts = [
            templates['high'],
            templates['medium'],
          templates['low'],
            templates['neutral']
        ]
    elif level == 'medium':
        # 中等强度：medium > high > low > neutral
        ranked_texts = [
         templates['medium'],
         templates['high'],
            templates['low'],
            templates['neutral']
        ]
    elif level == 'low':
        # 低强度：low > medium > neutral > high
        ranked_texts = [
            templates['low'],
            templates['medium'],
            templates['neutral'],
            templates['high']
        ]
    else:
        # 中性：neutral > low > medium > high
        ranked_texts = [
            templates['neutral'],
          templates['low'],
            templates['medium'],
            templates['high']
        ]
    
    return ranked_texts


def build_preference_dataset(csv_path, num_samples=1000, random_seed=42):
    """
    构建完整的偏好数据集
    
    Args:
        csv_path: 处理后的FER2013 CSV路径
        num_samples: 目标样本数
        random_seed: 随机种子
    
    Returns:
        preference_data: 偏好数据列表
    """
    print(f"\n读取数据: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # 每个情感类别均匀采样
    emotions = df['emotion_name'].unique()
    samples_per_emotion = num_samples // len(emotions)
    
    print(f"每个情感采样 {samples_per_emotion} 个")
    
    preference_data = []
    
    for emotion in emotions:
        emotion_df = df[df['emotion_name'] == emotion]
        
        # 采样
        n_sample = min(samples_per_emotion, len(emotion_df))
        samples = emotion_df.sample(n=n_sample, random_state=random_seed)
     
        for idx, row in samples.iterrows():
            intensity = row['intensity']
            
       # 构建排序描述
            ranked_texts = build_ranked_texts(emotion, intensity)
            
            preference_data.append({
                'image_id': int(idx),
              'pixels': row['pixels'],
                'emotion': emotion,
            'intensity': float(intensity),
                'intensity_level': get_intensity_level(intensity),
           'ranked_texts': ranked_texts
            })
    
    print(f"[OK] 构建完成！共 {len(preference_data)} 个样本")
    return preference_data


def split_and_save(preference_data, output_dir, train_ratio=0.8, val_ratio=0.1):
    """
    划分数据集并保存
    
    Args:
        preference_data: 偏好数据列表
      output_dir: 输出目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 划分
    train_data, temp_data = train_test_split(
        preference_data,
        test_size=1-train_ratio,
        random_state=42,
        stratify=[d['emotion'] for d in preference_data]
    )
    
    val_size = val_ratio / (1 - train_ratio)
    val_data, test_data = train_test_split(
        temp_data,
        test_size=1-val_size,
        random_state=42,
        stratify=[d['emotion'] for d in temp_data]
    )
    
    # 保存
    splits = {
        'preference_train.json': train_data,
        'preference_val.json': val_data,
        'preference_test.json': test_data
    }
    
    for filename, data in splits.items():
        output_path = output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[OK] 保存 {filename}: {len(data)} 个样本")


def print_statistics(preference_data):
    """打印数据集统计信息"""
    print("\n" + "=" * 50)
    print("数据集统计")
    print("=" * 50)
    
    # 情感分布
    emotion_counts = {}
    intensity_counts = {'high': 0, 'medium': 0, 'low': 0, 'neutral': 0}
    
    for item in preference_data:
        emotion = item['emotion']
        level = item['intensity_level']
        
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        intensity_counts[level] += 1
    
    print("\n情感分布:")
    for emotion, count in sorted(emotion_counts.items()):
        print(f"  {emotion}: {count}")
    
    print("\n强度等级分布:")
    for level, count in intensity_counts.items():
      print(f"  {level}: {count}")


def main():
    """主函数"""
    print("\n" + "=" * 50)
    print("构建有序偏好数据集")
    print("=" * 50)
    
    # 构建偏好数据集
    preference_data = build_preference_dataset(
     csv_path='./data/processed/fer2013_processed.csv',
        num_samples=1000
    )
    
    # 打印统计信息
    print_statistics(preference_data)
    
    # 划分并保存
    print("\n划分数据集...")
    split_and_save(
        preference_data,
        output_dir='./data/processed',
        train_ratio=0.8,
    val_ratio=0.1
    )
    
    # 打印示例
    print("\n" + "=" * 50)
    print("数据样例")
    print("=" * 50)
    sample = preference_data[0]
    print(f"\n图片ID: {sample['image_id']}")
    print(f"情感: {sample['emotion']}")
    print(f"强度值: {sample['intensity']:.3f}")
    print(f"强度等级: {sample['intensity_level']}")
    print(f"\n排序描述:")
    for i, text in enumerate(sample['ranked_texts']):
      print(f"  [{i+1}] {text}")
  
    print("\n" + "=" * 50)
    print("偏好数据集构建完成！")
    print("=" * 50)
    print("\n下一步：运行 'python training/train_lpo.py' 开始训练")


if __name__ == "__main__":
    main()
