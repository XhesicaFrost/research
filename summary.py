import json
import os
from datetime import datetime
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from datasets import Dataset

def load_summarization_model():
    """
    加载用于总结的轻量级模型
    """
    # 尝试不同的轻量级总结模型
    models_to_try = [
        "Qwen/Qwen2.5-0.5B-Instruct",        # Qwen指令模型，适合问题导向总结
        "sshleifer/distilbart-cnn-12-6",     # 轻量级版本，优先使用
        "t5-small",                          # 非常轻量
        "facebook/bart-large-cnn",           # 高质量但较大
        "google/pegasus-xsum",               # 专门用于摘要
    ]
    
    for model_name in models_to_try:
        try:
            print(f"正在尝试加载模型: {model_name}")
            
            if "qwen" in model_name.lower():
                # Qwen模型需要特殊处理
                summarizer = pipeline(
                    "text-generation",
                    model=model_name,
                    device=0 if torch.cuda.is_available() else -1,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    trust_remote_code=True,  # Qwen模型需要这个参数
                    batch_size=4  # Qwen模型使用较小的批处理
                )
            elif "t5" in model_name.lower():
                # T5模型需要特殊处理
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                summarizer = pipeline(
                    "summarization",
                    model=model,
                    tokenizer=tokenizer,
                    device=0 if torch.cuda.is_available() else -1,
                    batch_size=8  # 批处理大小
                )
            else:
                # 其他模型使用pipeline直接加载
                summarizer = pipeline(
                    "summarization",
                    model=model_name,
                    device=0 if torch.cuda.is_available() else -1,
                    batch_size=8  # 批处理大小
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
            device=0 if torch.cuda.is_available() else -1,
            batch_size=8
        )
        return summarizer, "gpt2-fallback"
    except Exception as e:
        print(f"备用模型也加载失败: {e}")
        return None, None

def generate_summary_with_qwen(question, text, summarizer):
    """
    使用Qwen模型生成针对问题的总结
    """
    try:
        # 构建Qwen格式的消息，专门用于总结
        messages = [
            {
                "role": "user", 
                "content": f"""Based on the following context, create a focused summary that contains information specifically relevant to answering the question. 

Question: {question}

Context: {text}

Please provide a concise summary (50-100 words) that highlights the key information needed to answer the question. Focus on facts, dates, names, and relationships mentioned in the context:"""
            }
        ]
        
        response = summarizer(
            messages,
            max_new_tokens=200,  # 给总结足够的空间
            num_return_sequences=1,
            temperature=0.3,  # 较低温度获得更聚焦的总结
            do_sample=True,
            pad_token_id=summarizer.tokenizer.eos_token_id,
            eos_token_id=summarizer.tokenizer.eos_token_id
        )
        
        print(f"Debug - Qwen总结原始响应: {response}")  # 调试信息
        
        # 提取生成的内容
        summary = ""
        if isinstance(response, list) and len(response) > 0:
            generated_text = response[0].get('generated_text', '')
            
            # 如果返回的是消息格式
            if isinstance(generated_text, list) and len(generated_text) > 1:
                # 获取助手的回复
                assistant_reply = generated_text[-1].get('content', '')
                summary = assistant_reply.strip()
            else:
                # 如果是字符串格式，直接处理
                summary = str(generated_text).strip()
        
        # 处理Qwen的特殊标记，参考answer.py的处理逻辑
        if summary:
            import re
            
            # 处理<think>标记
            if "<think>" in summary:
                if "</think>" in summary:
                    # 提取</think>后的内容作为最终总结
                    summary = summary.split("</think>")[-1].strip()
                else:
                    # 如果没有闭合标签，尝试从<think>后提取总结
                    think_part = summary.split("<think>")[-1]
                    # 查找总结内容
                    lines = think_part.split('\n')
                    summary_lines = []
                    for line in lines:
                        line = line.strip()
                        # 跳过明显的思考过程行
                        if line and not line.lower().startswith(('let me', 'i need to', 'looking at', 'the question', 'based on')):
                            # 寻找包含实际信息的行
                            if any(word in line.lower() for word in ['magazine', 'published', 'started', 'founded', 'year']):
                                summary_lines.append(line)
                    
                    if summary_lines:
                        summary = ' '.join(summary_lines[:3])  # 最多3行
                    else:
                        # 如果没找到合适的行，使用关键信息提取
                        return extract_key_info_for_question(question, text)
            
            # 清理总结中的多余内容
            if "Context:" in summary:
                summary = summary.split("Context:")[-1]
            if "Question:" in summary:
                summary = summary.split("Question:")[-1]
            if "Summary:" in summary:
                summary = summary.split("Summary:")[-1].strip()
            
            # 移除常见的总结前缀
            prefixes_to_remove = [
                "Here is a summary", "Here's a summary", "The summary is", 
                "Based on the context", "According to", "The key information"
            ]
            for prefix in prefixes_to_remove:
                if summary.lower().startswith(prefix.lower()):
                    summary = summary[len(prefix):].strip()
                    if summary.startswith(":"):
                        summary = summary[1:].strip()
                    break
            
            # 清理格式
            summary = summary.replace('\\n', ' ').replace('\n', ' ')
            summary = ' '.join(summary.split())  # 标准化空格
            
            # 确保总结长度合适
            words = summary.split()
            if len(words) > 200:  # 如果太长，截取到80词
                summary = ' '.join(words[:200]) + "..."
            elif len(words) < 10:  # 如果太短，可能质量不好
                print(f"Debug - Qwen总结太短: '{summary}', 使用备用方法")
                return extract_key_info_for_question(question, text)
            
            print(f"Debug - Qwen最终总结: '{summary}'")
            
            return summary.strip()
        
        # 如果没有获得有效总结，使用备用方法
        return extract_key_info_for_question(question, text)
        
    except Exception as e:
        print(f"Qwen生成总结时出错: {e}")
        import traceback
        traceback.print_exc()
        # 如果出错，使用关键信息提取
        return extract_key_info_for_question(question, text)

def generate_summary_with_qwen_fallback(question, text, summarizer):
    """
    Qwen模型的备用总结方案
    """
    try:
        # 使用更简单的prompt格式
        prompt = f"""Question: {question}

Context: {text}

Create a brief summary focusing on information relevant to the question:"""
        
        response = summarizer(
            prompt,
            max_new_tokens=120,
            num_return_sequences=1,
            temperature=0.3,
            do_sample=True,
            pad_token_id=summarizer.tokenizer.eos_token_id,
            eos_token_id=summarizer.tokenizer.eos_token_id
        )
        
        generated_text = response[0]['generated_text']
        
        # 提取总结部分
        if "Create a brief summary" in generated_text:
            summary = generated_text.split("Create a brief summary")[-1]
            if ":" in summary:
                summary = summary.split(":", 1)[1].strip()
        else:
            summary = generated_text.replace(prompt, "").strip()
        
        # 处理think标记
        import re
        if "<think>" in summary:
            if "</think>" in summary:
                summary = summary.split("</think>")[-1].strip()
            else:
                # 从think内容中提取关键信息
                think_content = summary.split("<think>")[-1]
                lines = think_content.split('\n')
                for line in lines:
                    line = line.strip()
                    if line and len(line.split()) > 5 and len(line.split()) < 30:
                        summary = line
                        break
                else:
                    return extract_key_info_for_question(question, text)
        
        # 清理总结
        summary = summary.split('\n')[0].strip()
        summary = re.sub(r'\([^)]*\)', '', summary).strip()
        
        words = summary.split()
        if len(words) > 60:
            summary = " ".join(words[:60]) + "..."
        
        print(f"Debug - Qwen备用总结: '{summary}'")
        
        return summary if summary else extract_key_info_for_question(question, text)
        
    except Exception as e:
        print(f"Qwen备用总结方案失败: {e}")
        return extract_key_info_for_question(question, text)

def generate_summaries_batch(batch_inputs, summarizer, model_name, batch_size=8):
    """
    批量生成针对问题的总结，并输出实时进度
    """
    if not batch_inputs or not summarizer:
        return []
    
    summaries = []
    
    try:
        print(f"\n{'='*60}")
        print(f"开始批量生成总结")
        print(f"总问题数: {len(batch_inputs)}")
        print(f"批处理大小: {batch_size}")
        print(f"使用模型: {model_name}")
        print(f"{'='*60}")
        
        if "qwen" in model_name.lower():
            # Qwen模型处理，逐个处理以确保质量
            print("📝 使用Qwen模型逐个处理...")
            for idx, item in enumerate(batch_inputs):
                question = item['question']
                text = item['text']
                
                print(f"🔄 处理第 {idx + 1}/{len(batch_inputs)} 个问题...")
                print(f"   问题: {question[:80]}...")
                
                # 使用主要的Qwen总结方法
                summary = generate_summary_with_qwen(question, text, summarizer)
                
                # 如果主要方法失败，尝试备用方法
                if not summary or len(summary.strip()) < 10:
                    print("   🔄 主要方法结果不佳，尝试备用方法...")
                    summary = generate_summary_with_qwen_fallback(question, text, summarizer)
                
                summaries.append(summary)
                
                print(f"   ✅ 生成总结: {summary[:100]}...")
                
                # 每处理5个显示进度
                if (idx + 1) % 5 == 0 or (idx + 1) == len(batch_inputs):
                    progress = (idx + 1) / len(batch_inputs) * 100
                    print(f"📊 Qwen处理进度: {idx + 1}/{len(batch_inputs)} ({progress:.1f}%)")
                    print("-" * 40)
        
        elif "gpt2-fallback" in model_name:
            # GPT2备用方案，逐个处理
            print("📝 使用GPT2备用方案，逐个处理...")
            for idx, item in enumerate(batch_inputs):
                question = item['question']
                text = item['text']
                summary = generate_question_focused_summary(question, [text], summarizer, model_name)
                summaries.append(summary)
                
                # 每处理1个就显示进度
                if (idx + 1) % 1 == 0:
                    progress = (idx + 1) / len(batch_inputs) * 100
                    print(f"🔄 第 {idx + 1}/{len(batch_inputs)} 个总结完成 ({progress:.1f}%) - {summary[:50]}...")
        
        elif "t5" in model_name.lower():
            # T5模型分批处理，改进prompt
            print("🤖 使用T5模型分批处理...")
            
            # 准备输入数据 - 改进prompt设计
            print("准备输入数据...")
            input_texts = []
            for idx, item in enumerate(batch_inputs):
                # 改进的prompt，更明确地指向问题
                question = item['question']
                text = item['text']
                
                # 根据问题类型设计不同的prompt
                if "which" in question.lower() and ("first" in question.lower() or "started" in question.lower()):
                    # 对于比较类问题，强调时间比较
                    input_text = f"Compare the start dates mentioned in the text to answer: {question}. Context: {text}"
                elif "when" in question.lower():
                    input_text = f"Extract dates and timeline information to answer: {question}. Context: {text}"
                elif "who" in question.lower():
                    input_text = f"Identify people and their roles to answer: {question}. Context: {text}"
                else:
                    input_text = f"Extract key information specifically relevant to: {question}. Context: {text}"
                
                input_texts.append(input_text)
                
                if (idx + 1) % 100 == 0:
                    print(f"   输入准备进度: {idx + 1}/{len(batch_inputs)}")
            
            print(f"✅ 输入数据准备完成，开始分批推理...")
            
            # 分批处理，每批都显示进度
            total_batches = (len(input_texts) + batch_size - 1) // batch_size
            
            for batch_idx in range(0, len(input_texts), batch_size):
                end_idx = min(batch_idx + batch_size, len(input_texts))
                batch_texts = input_texts[batch_idx:end_idx]
                current_batch_num = batch_idx // batch_size + 1
                
                print(f"🔄 处理第 {current_batch_num}/{total_batches} 批 (项目 {batch_idx+1}-{end_idx})...")
                
                try:
                    # 处理当前批次
                    batch_results = summarizer(
                        batch_texts,
                        max_length=120,
                        min_length=20,
                        do_sample=False
                    )
                    
                    # 提取总结并添加到结果列表
                    batch_summaries = [result['summary_text'] for result in batch_results]
                    summaries.extend(batch_summaries)
                    
                    # 显示当前批次的结果示例
                    if batch_summaries:
                        print(f"   ✅ 批次 {current_batch_num} 完成，示例: {batch_summaries[0][:60]}...")
                    
                    # 显示总体进度
                    overall_progress = len(summaries) / len(input_texts) * 100
                    print(f"   📊 总体进度: {len(summaries)}/{len(input_texts)} ({overall_progress:.1f}%)")
                    
                except Exception as e:
                    print(f"   ❌ 批次 {current_batch_num} 处理失败: {e}")
                    # 为失败的批次添加空总结
                    summaries.extend([""] * len(batch_texts))
            
            print(f"✅ T5模型分批处理完成，共生成 {len(summaries)} 个总结")
        
        else:
            # 标准总结模型分批处理 - 改进prompt
            print(f"📊 使用标准总结模型分批处理...")
            
            # 准备输入数据 - 改进prompt设计
            print("准备输入数据...")
            input_texts = []
            for idx, item in enumerate(batch_inputs):
                question = item['question']
                text = item['text']
                
                # 改进的prompt设计，更聚焦于问题
                if "which" in question.lower() and ("first" in question.lower() or "started" in question.lower()):
                    # 对于比较类问题，强调找出两个实体的开始时间
                    input_text = f"Question asks which started first. Find the start dates of each mentioned entity. Question: {question}. Information: {text}"
                elif "when" in question.lower():
                    input_text = f"Question asks about timing. Extract all dates and time information. Question: {question}. Information: {text}"
                elif "who" in question.lower():
                    input_text = f"Question asks about people. Identify all people mentioned and their roles. Question: {question}. Information: {text}"
                elif "where" in question.lower():
                    input_text = f"Question asks about location. Extract all places and locations mentioned. Question: {question}. Information: {text}"
                elif "what" in question.lower():
                    input_text = f"Question asks for definition or description. Extract key characteristics and descriptions. Question: {question}. Information: {text}"
                else:
                    input_text = f"Extract information specifically needed to answer this question: {question}. Available information: {text}"
                
                # 限制输入长度
                if len(input_text) > 1000:
                    input_text = input_text[:1000]
                input_texts.append(input_text)
                
                if (idx + 1) % 100 == 0:
                    print(f"   输入准备进度: {idx + 1}/{len(batch_inputs)}")
            
            print(f"✅ 输入数据准备完成，开始分批推理...")
            
            # 分批处理，每批都显示进度
            total_batches = (len(input_texts) + batch_size - 1) // batch_size
            
            for batch_idx in range(0, len(input_texts), batch_size):
                end_idx = min(batch_idx + batch_size, len(input_texts))
                batch_texts = input_texts[batch_idx:end_idx]
                current_batch_num = batch_idx // batch_size + 1
                
                print(f"🔄 处理第 {current_batch_num}/{total_batches} 批 (项目 {batch_idx+1}-{end_idx})...")
                
                try:
                    # 处理当前批次
                    batch_results = summarizer(
                        batch_texts,
                        max_length=150,  # 稍微增加长度以容纳比较信息
                        min_length=30,
                        do_sample=False
                    )
                    
                    # 提取总结并添加到结果列表
                    batch_summaries = [result['summary_text'] for result in batch_results]
                    summaries.extend(batch_summaries)
                    
                    # 显示当前批次的结果示例
                    if batch_summaries:
                        print(f"   ✅ 批次 {current_batch_num} 完成，示例: {batch_summaries[0][:80]}...")
                    
                    # 显示总体进度
                    overall_progress = len(summaries) / len(input_texts) * 100
                    print(f"   📊 总体进度: {len(summaries)}/{len(input_texts)} ({overall_progress:.1f}%)")
                    print(f"   ⏱️  已完成 {len(summaries)} 个总结")
                    
                    # 每完成几个批次显示更详细的信息
                    if current_batch_num % 5 == 0:
                        print(f"   🎯 里程碑: 已完成 {current_batch_num} 个批次，继续处理...")
                        print(f"   📈 效率: 平均每批处理 {len(batch_summaries)} 个项目")
                        print("-" * 40)
                    
                except Exception as e:
                    print(f"   ❌ 批次 {current_batch_num} 处理失败: {e}")
                    # 尝试单个处理这个批次
                    print(f"   🔄 尝试单个处理批次 {current_batch_num}...")
                    batch_summaries = []
                    for single_text in batch_texts:
                        try:
                            single_result = summarizer(
                                single_text,
                                max_length=150,
                                min_length=30,
                                do_sample=False
                            )
                            batch_summaries.append(single_result[0]['summary_text'])
                        except:
                            batch_summaries.append("")
                    
                    summaries.extend(batch_summaries)
                    print(f"   ✅ 批次 {current_batch_num} 单个处理完成")
            
            print(f"✅ 标准模型分批处理完成，共生成 {len(summaries)} 个总结")
        
        # 清理和验证总结（也显示进度）
        print(f"\n{'='*40}")
        print("🧹 开始清理和验证总结...")
        print(f"{'='*40}")
        
        cleaned_summaries = []
        poor_quality_count = 0
        
        for i, summary in enumerate(summaries):
            summary = summary.strip()
            
            # 如果总结质量不好，使用关键信息提取
            if len(summary) < 10 or summary.lower().startswith(("the", "this", "it")):
                question = batch_inputs[i]['question']
                text = batch_inputs[i]['text']
                summary = extract_key_info_for_question(question, text)
                poor_quality_count += 1
                
                if poor_quality_count <= 3:  # 只显示前3个质量问题
                    print(f"⚠️  第{i+1}个总结质量不佳，使用关键信息提取")
            
            cleaned_summaries.append(summary)
            
            # 每处理完50条显示一次清理进度
            if (i + 1) % 50 == 0 or (i + 1) == len(summaries):
                progress = (i + 1) / len(summaries) * 100
                print(f"🧹 清理进度: {i + 1}/{len(summaries)} ({progress:.1f}%)")
        
        print(f"\n{'='*50}")
        print(f"✅ 总结生成完成！")
        print(f"📊 统计信息:")
        print(f"   - 总问题数: {len(batch_inputs)}")
        print(f"   - 成功生成: {len(cleaned_summaries)}")
        print(f"   - 质量问题: {poor_quality_count}")
        print(f"   - 成功率: {(len(cleaned_summaries)-poor_quality_count)/len(cleaned_summaries)*100:.1f}%")
        print(f"{'='*50}")
        
        return cleaned_summaries
        
    except Exception as e:
        print(f"❌ 批量生成问题导向总结时出错: {e}")
        import traceback
        traceback.print_exc()
        
        # 降级到逐个处理（显示每个处理进度）
        print(f"\n{'='*40}")
        print("🆘 降级到逐个处理模式...")
        print(f"{'='*40}")
        
        summaries = []
        error_count = 0
        
        for idx, item in enumerate(batch_inputs):
            try:
                question = item['question']
                text = item['text']
                summary = extract_key_info_for_question(question, text)
                summaries.append(summary)
                
                # 每个都显示进度
                progress = (idx + 1) / len(batch_inputs) * 100
                print(f"🔄 降级模式: {idx + 1}/{len(batch_inputs)} ({progress:.1f}%) - {summary[:40]}...")
                
                # 每10个显示一个里程碑
                if (idx + 1) % 10 == 0:
                    print(f"📈 里程碑: 已完成 {idx + 1} 个，继续处理...")
                    
            except Exception as e2:
                error_count += 1
                print(f"❌ 处理第{idx+1}个问题时出错: {e2}")
                summaries.append("")
        
        print(f"\n{'='*40}")
        print(f"✅ 降级模式完成")
        print(f"📊 统计信息:")
        print(f"   - 处理总数: {len(batch_inputs)}")
        print(f"   - 成功处理: {len(summaries) - error_count}")
        print(f"   - 错误数量: {error_count}")
        print(f"   - 成功率: {(len(summaries) - error_count)/len(summaries)*100:.1f}%")
        print(f"{'='*40}")
        
        return summaries

def extract_key_info_for_question(question, text):
    """
    基于问题类型提取关键信息作为总结
    """
    import re
    
    question_lower = question.lower()
    
    # 特别处理比较类问题
    if "which" in question_lower and ("first" in question_lower or "started" in question_lower):
        # 提取所有年份和实体
        sentences = text.split('.')
        relevant_info = []
        
        # 查找年份信息
        for sentence in sentences:
            # 查找四位数年份
            years = re.findall(r'\b(1[89]\d{2}|20\d{2})\b', sentence)
            if years:
                relevant_info.append(sentence.strip())
        
        # 如果找到时间信息，返回这些句子
        if relevant_info:
            summary = '. '.join(relevant_info[:3])  # 最多三句
        else:
            # 如果没找到明确年份，查找包含"started"、"founded"、"began"等词的句子
            for sentence in sentences:
                if any(word in sentence.lower() for word in ['started', 'founded', 'began', 'established', 'published']):
                    relevant_info.append(sentence.strip())
            summary = '. '.join(relevant_info[:2])
    
    elif "who" in question_lower:
        # 提取人名相关的句子
        sentences = text.split('.')
        relevant_sentences = []
        for sentence in sentences:
            if any(word[0].isupper() and len(word) > 2 for word in sentence.split()):
                relevant_sentences.append(sentence.strip())
        summary = '. '.join(relevant_sentences[:2])
    
    elif "when" in question_lower or "year" in question_lower:
        # 提取包含时间信息的句子
        sentences = text.split('.')
        relevant_sentences = []
        for sentence in sentences:
            if re.search(r'\b(19|20)\d{2}\b|\b(January|February|March|April|May|June|July|August|September|October|November|December)\b', sentence):
                relevant_sentences.append(sentence.strip())
        summary = '. '.join(relevant_sentences[:2])
    
    elif "where" in question_lower or "city" in question_lower or "country" in question_lower:
        # 提取地点相关信息
        sentences = text.split('.')
        relevant_sentences = []
        for sentence in sentences:
            # 寻找地点标识词
            location_words = ['in', 'at', 'from', 'city', 'country', 'located', 'based']
            if any(word in sentence.lower() for word in location_words):
                relevant_sentences.append(sentence.strip())
        summary = '. '.join(relevant_sentences[:2])
    
    elif "what" in question_lower:
        # 提取包含定义或描述的句子
        sentences = text.split('.')
        relevant_sentences = []
        for sentence in sentences:
            # 寻找定义性词汇
            definition_words = ['is', 'was', 'are', 'were', 'means', 'refers', 'describes']
            if any(word in sentence.lower().split() for word in definition_words):
                relevant_sentences.append(sentence.strip())
        summary = '. '.join(relevant_sentences[:2])
    
    else:
        # 默认情况：取前两句
        sentences = text.split('.')
        summary = '. '.join(sentences[:2])
    
    # 限制长度
    if len(summary) > 300:
        summary = summary[:300] + "..."
    
    return summary.strip() if summary.strip() else text[:200]

def prepare_batch_data(data_list, max_input_length=800):
    """
    准备批处理数据，包含问题信息
    """
    batch_inputs = []
    indices = []
    
    for i, data in enumerate(data_list):
        question = data.get('question', '')
        related = data.get('related', [])
        
        if related and question:
            # 合并相关段落
            combined_text = " ".join(related).strip()
            
            # 限制输入长度
            if len(combined_text) > max_input_length:
                combined_text = combined_text[:max_input_length]
            
            batch_inputs.append({
                'question': question,
                'text': combined_text
            })
            indices.append(i)
    
    return batch_inputs, indices


def process_extraction_file(input_file, batch_size=16):
    """
    处理提取文件，为每行添加基于问题的summary字段
    """
    # 加载总结模型
    print("正在加载总结模型...")
    summarizer, model_name = load_summarization_model()
    
    if not summarizer:
        print("无法加载任何模型，退出程序")
        return
    
    print(f"使用模型: {model_name}")
    print(f"批处理大小: {batch_size}")
    print("模式: 基于问题生成针对性总结")
    
    # 生成输出文件名
    timestamp = datetime.now().strftime("%Y%m%d")
    input_filename = os.path.basename(input_file)
    output_filename = f"{timestamp}_summary_{input_filename}"
    output_dir = "/home/xhesica/research/outputs"
    output_file = os.path.join(output_dir, output_filename)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 读取所有数据
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 解析JSON数据
        data_list = []
        for i, line in enumerate(lines):
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    data_list.append(data)
                except json.JSONDecodeError as e:
                    print(f"第{i+1}行JSON解析错误: {e}")
                    continue
        
        total_items = len(data_list)
        print(f"成功解析 {total_items} 条记录，开始批量处理...")
        
        # 准备批处理数据
        batch_inputs, indices = prepare_batch_data(data_list)
        
        if not batch_inputs:
            print("没有找到包含问题和相关文本的记录")
            return
        
        print(f"准备为 {len(batch_inputs)} 个问题生成针对性总结...")
        
        # 批量生成总结
        summaries = generate_summaries_batch(batch_inputs, summarizer, model_name, batch_size)
        
        # 将总结结果映射回原始数据
        for summary, idx in zip(summaries, indices):
            data_list[idx]['summary'] = summary
        
        # 为没有相关文本或问题的记录设置空总结
        for data in data_list:
            if 'summary' not in data:
                data['summary'] = ""
        
        # 保存处理后的数据
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in data_list:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
        
        print(f"基于问题的总结处理完成！共处理 {len(data_list)} 条记录")
        print(f"其中 {len(summaries)} 条记录生成了针对性总结")
        print(f"输出文件: {output_file}")
        
        # 显示一些示例结果
        print("\n示例总结结果:")
        count = 0
        for item in data_list:
            if item.get('summary') and item.get('question') and count < 3:
                print(f"问题: {item.get('question', '')[:100]}...")
                print(f"针对性总结: {item.get('summary', '')[:200]}...")
                print("-" * 50)
                count += 1
        
    except FileNotFoundError:
        print(f"文件未找到: {input_file}")
        return
    except Exception as e:
        print(f"处理文件时出错: {e}")
        import traceback
        traceback.print_exc()

def main():
    """
    主函数
    """
    input_file = "/home/xhesica/research/outputs/20250607_extract_20250606_answer_answer_train_limit2000.json"
    
    # 可以调整批处理大小
    batch_size = 16  # 根据GPU内存调整
    
    print(f"基于问题的Summary处理配置:")
    print(f"- 输入文件: {input_file}")
    print(f"- 批处理大小: {batch_size}")
    print(f"- 使用GPU: {torch.cuda.is_available()}")
    print(f"- 特色: 根据每个问题生成针对性总结")
    print("-" * 50)
    
    process_extraction_file(input_file, batch_size=batch_size)

if __name__ == "__main__":
    main()