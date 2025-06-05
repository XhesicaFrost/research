import json
import os
from datetime import datetime
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

def load_summarization_model():
    """
    加载用于总结的轻量级模型
    """
    # 尝试不同的轻量级总结模型
    models_to_try = [
        "facebook/bart-large-cnn",  # 高质量但较大
        "sshleifer/distilbart-cnn-12-6",  # 轻量级版本
        "t5-small",  # 非常轻量
        "google/pegasus-xsum",  # 专门用于摘要
    ]
    
    for model_name in models_to_try:
        try:
            print(f"正在尝试加载模型: {model_name}")
            
            if "t5" in model_name.lower():
                # T5模型需要特殊处理
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                summarizer = pipeline(
                    "summarization",
                    model=model,
                    tokenizer=tokenizer,
                    device=0 if torch.cuda.is_available() else -1
                )
            else:
                # 其他模型使用pipeline直接加载
                summarizer = pipeline(
                    "summarization",
                    model=model_name,
                    device=0 if torch.cuda.is_available() else -1
                )
            
            print(f"成功加载模型: {model_name}")
            return summarizer, model_name
            
        except Exception as e:
            print(f"模型 {model_name} 加载失败: {e}")
            continue
    
    # 如果所有模型都失败，使用备用方案
    print("使用备用的文本生成模型进行总结...")
    try:
        summarizer = pipeline(
            "text-generation",
            model="gpt2",
            device=0 if torch.cuda.is_available() else -1
        )
        return summarizer, "gpt2-fallback"
    except Exception as e:
        print(f"备用模型也加载失败: {e}")
        return None, None

def generate_summary_with_model(related_passages, summarizer, model_name):
    """
    使用模型生成总结
    """
    if not related_passages or not summarizer:
        return ""
    
    # 合并所有相关段落
    combined_text = " ".join(related_passages).strip()
    
    # 限制输入长度（避免超出模型限制）
    max_input_length = 1000
    if len(combined_text) > max_input_length:
        combined_text = combined_text[:max_input_length]
    
    try:
        if "gpt2-fallback" in model_name:
            # 使用GPT2作为备用方案
            prompt = f"Summarize the following text: {combined_text}\nSummary:"
            result = summarizer(
                prompt,
                max_length=len(prompt.split()) + 50,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=summarizer.tokenizer.eos_token_id
            )
            summary = result[0]['generated_text'].split("Summary:")[-1].strip()
        
        elif "t5" in model_name.lower():
            # T5模型需要特殊的前缀
            input_text = f"summarize: {combined_text}"
            result = summarizer(
                input_text,
                max_length=150,
                min_length=30,
                do_sample=False
            )
            summary = result[0]['summary_text']
        
        else:
            # 标准的总结模型
            result = summarizer(
                combined_text,
                max_length=150,
                min_length=30,
                do_sample=False
            )
            summary = result[0]['summary_text']
        
        return summary.strip()
        
    except Exception as e:
        print(f"生成总结时出错: {e}")
        # 降级到简单文本处理
        return combined_text[:200] + "..." if len(combined_text) > 200 else combined_text

def process_extraction_file(input_file):
    """
    处理提取文件，为每行添加summary字段
    """
    # 加载总结模型
    print("正在加载总结模型...")
    summarizer, model_name = load_summarization_model()
    
    if not summarizer:
        print("无法加载任何模型，退出程序")
        return
    
    print(f"使用模型: {model_name}")
    
    # 生成输出文件名
    timestamp = datetime.now().strftime("%Y%m%d")
    input_filename = os.path.basename(input_file)
    output_filename = f"{timestamp}_summary_{input_filename}"
    output_file = os.path.join("outputs", output_filename)
    
    # 确保输出目录存在
    os.makedirs("outputs", exist_ok=True)
    
    processed_data = []
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        total_lines = len(lines)
        print(f"开始处理 {total_lines} 条记录...")
        
        for i, line in enumerate(lines):
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    
                    # 获取related字段
                    related = data.get('related', [])
                    
                    # 生成总结
                    print(f"处理进度: {i+1}/{total_lines} - 生成总结中...")
                    summary = generate_summary_with_model(related, summarizer, model_name)
                    
                    # 添加summary字段
                    data['summary'] = summary
                    
                    processed_data.append(data)
                    
                except json.JSONDecodeError as e:
                    print(f"第{i+1}行JSON解析错误: {e}")
                    continue
                except Exception as e:
                    print(f"处理第{i+1}行时出错: {e}")
                    continue
    
    except FileNotFoundError:
        print(f"文件未找到: {input_file}")
        return
    
    # 保存处理后的数据
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in processed_data:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
        
        print(f"处理完成！共处理 {len(processed_data)} 条记录")
        print(f"输出文件: {output_file}")
        
    except Exception as e:
        print(f"保存文件时出错: {e}")

def main():
    """
    主函数
    """
    input_file = "/home/xhesica/research/outputs/20250605_extract_20250605_answer_answer_train_limit1000.json"
    process_extraction_file(input_file)

if __name__ == "__main__":
    main()