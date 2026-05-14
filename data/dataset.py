"""
情感偏好数据集类
用于加载和处理FER2013的有序偏好数据
"""
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import json


class EmotionPreferenceDataset(Dataset):
    """情感偏好数据集"""
    
    def __init__(self, json_path, processor=None, image_size=224, split='train'):
        """
     Args:
            json_path: 偏好数据JSON路径
            processor: 模型处理器（如LLaVA processor）
            image_size: 图片尺寸
            split: 数据集划分（train/val/test）
        """
        print(f"加载数据集: {json_path}")
      with open(json_path, 'r', encoding='utf-8') as f:
          self.data = json.load(f)
        
        self.processor = processor
        self.image_size = image_size
    self.split = split
      
        print(f"[OK] {split}集加载完成，共 {len(self.data)} 个样本")
    
    def __len__(self):
        return len(self.data)
    
    def _pixels_to_image(self, pixels_str):
        """将像素字符串转换为PIL图像"""
      # FER2013是48x48灰度图
        pixels = np.array([int(p) for p in pixels_str.split()], dtype=np.uint8)
        pixels = pixels.reshape(48, 48)
        
        # 转换为RGB并调整大小
        image = Image.fromarray(pixels).convert('RGB')
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        
        return image
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 加载图像
        image = self._pixels_to_image(item['pixels'])
        
     # 获取排序文本
        ranked_texts = item['ranked_texts']
        
        # 如果有processor，进行处理
        if self.processor is not None:
       # 处理图像
            image_inputs = self.processor(
            images=image,
                return_tensors="pt"
            )
            
            # 处理每个文本
         text_inputs = []
        for text in ranked_texts:
        text_input = self.processor.tokenizer(
                    text,
                    return_tensors="pt",
             padding='max_length',
            truncation=True,
                  max_length=128
              )
                text_inputs.append(text_input)
          
       return {
                'pixel_values': image_inputs['pixel_values'].squeeze(0),
                'input_ids': torch.stack([t['input_ids'].squeeze(0) for t in text_inputs]),
                'attention_mask': torch.stack([t['attention_mask'].squeeze(0) for t in text_inputs]),
              'ranked_texts': ranked_texts,
             'emotion': item['emotion'],
                'intensity': item['intensity'],
                'intensity_level': item['intensity_level']
       }
        else:
          # 返回原始数据
            return {
           'image': image,
                'ranked_texts': ranked_texts,
             'emotion': item['emotion'],
            'intensity': item['intensity'],
           'intensity_level': item['intensity_level']
         }


def collate_fn(batch):
    """自定义batch collation"""
    pixel_values = torch.stack([b['pixel_values'] for b in batch])
    input_ids = torch.stack([b['input_ids'] for b in batch])
    attention_mask = torch.stack([b['attention_mask'] for b in batch])
    
    return {
      'pixel_values': pixel_values,  # [B, C, H, W]
        'input_ids': input_ids,  # [B, N, L]
        'attention_mask': attention_mask,  # [B, N, L]
     'ranked_texts': [b['ranked_texts'] for b in batch],
      'emotions': [b['emotion'] for b in batch],
        'intensities': torch.tensor([b['intensity'] for b in batch]),
        'intensity_levels': [b['intensity_level'] for b in batch]
    }


def get_dataloader(json_path, processor, batch_size=4, num_workers=4, shuffle=True, split='train'):
    """创建数据加载器"""
    dataset = EmotionPreferenceDataset(
        json_path=json_path,
        processor=processor,
        split=split
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
      num_workers=num_workers,
      collate_fn=collate_fn,
      pin_memory=True
    )
    
    return dataloader


if __name__ == "__main__":
    # 测试数据加载
    print("测试数据集加载...")
    
    dataset = EmotionPreferenceDataset(
        json_path='./data/processed/preference_train.json',
        processor=None,
     split='train'
    )
    
    print(f"\n数据集大小: {len(dataset)}")
    
    # 查看第一个样本
    sample = dataset[0]
    print(f"\n样本示例:")
    print(f"  情感: {sample['emotion']}")
    print(f"  强度: {sample['intensity']:.3f}")
    print(f"  强度等级: {sample['intensity_level']}")
  print(f"  图像尺寸: {sample['image'].size}")
    print(f"  排序文本数: {len(sample['ranked_texts'])}")
