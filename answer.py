import json
import os
from datetime import datetime
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

def load_qa_model():
    """
    加载用于问答的模型
    """
    # 尝试不同的模型，优先使用Qwen3-0.6B
    models_to_try = [
        #"Qwen/Qwen3-0.6B",  # 主要使用的Qwen模型
        #"meta-llama/Llama-3.2-1B",  # 备选模型
        "distilbert-base-cased-distilled-squad",  # 轻量级问答模型
        "gpt2",  # 最后备选
    ]
    
    for model_name in models_to_try:
        try:
            print(f"正在尝试加载模型: {model_name}")
            
            if "qwen" in model_name.lower():
                # 使用pipeline方式加载Qwen模型
                qa_pipeline = pipeline(
                    "text-generation",
                    model=model_name,
                    device=0 if torch.cuda.is_available() else -1,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
                print(f"成功加载Qwen模型: {model_name}")
                return qa_pipeline, model_name
            
            elif "llama" in model_name.lower():
                # 使用pipeline方式加载Llama模型
                qa_pipeline = pipeline(
                    "text-generation",
                    model=model_name,
                    device=0 if torch.cuda.is_available() else -1,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
                print(f"成功加载Llama模型: {model_name}")
                return qa_pipeline, model_name
            
            elif "distilbert" in model_name.lower() or "bert" in model_name.lower():
                # 使用专门的问答模型
                qa_pipeline = pipeline(
                    "question-answering",
                    model=model_name,
                    device=0 if torch.cuda.is_available() else -1
                )
                print(f"成功加载问答模型: {model_name}")
                return qa_pipeline, model_name
            
            else:
                # 其他文本生成模型
                qa_pipeline = pipeline(
                    "text-generation",
                    model=model_name,
                    device=0 if torch.cuda.is_available() else -1
                )
                print(f"成功加载模型: {model_name}")
                return qa_pipeline, model_name
            
        except Exception as e:
            print(f"模型 {model_name} 加载失败: {e}")
            continue
    
    print("所有模型都加载失败！")
    return None, None

def generate_answer_with_qwen(question, summary, qa_pipeline):
    """
    使用Qwen模型生成答案
    """
    try:
        # 构建Qwen格式的消息，在prompt中明确要求简洁回答
        messages = [
            {
                "role": "user", 
                "content": f"""Based on the following context, please answer the question with ONLY the essential information. Your answer should be 1-5 words maximum.

Context: {summary}

Question: {question}

Important: Give only the direct answer, no explanation, no thinking process, just the core answer in 1-5 words:"""
            }
        ]
        
        response = qa_pipeline(
            messages,
            max_new_tokens=200,  # 给足够的空间，不截断
            num_return_sequences=1,
            temperature=0.1,  # 低温度获得更确定的答案
            do_sample=True,
            pad_token_id=qa_pipeline.tokenizer.eos_token_id,
            eos_token_id=qa_pipeline.tokenizer.eos_token_id
        )
        
        print(f"Debug - Qwen原始响应: {response}")  # 调试信息
        
        # 提取生成的内容
        answer = ""
        if isinstance(response, list) and len(response) > 0:
            generated_text = response[0].get('generated_text', '')
            print(f"Debug - generated_text类型: {type(generated_text)}")
            
            # 如果返回的是消息格式
            if isinstance(generated_text, list) and len(generated_text) > 1:
                # 获取助手的回复
                assistant_reply = generated_text[-1].get('content', '')
                print(f"Debug - assistant_reply: {assistant_reply}")
                answer = assistant_reply.strip()
            else:
                # 如果是字符串格式，直接处理
                answer = str(generated_text).strip()
        
        # 处理Qwen的回答，但不强行截断
        if answer:
            import re
            
            # 如果包含<think>标记，提取思考过程后的答案
            if "<think>" in answer:
                # 查找</think>后的内容作为最终答案
                if "</think>" in answer:
                    answer = answer.split("</think>")[-1].strip()
                else:
                    # 如果没有闭合标签，尝试从<think>后提取简短答案
                    think_part = answer.split("<think>")[-1]
                    # 查找可能的答案模式
                    lines = think_part.split('\n')
                    for line in lines:
                        line = line.strip()
                        # 寻找简短的答案行（少于10个词）
                        if line and len(line.split()) <= 10 and not line.startswith(('The', 'Based', 'According', 'From')):
                            answer = line
                            break
                    else:
                        # 如果没找到合适的行，使用上下文提取
                        return extract_answer_from_context(question, summary)
            
            # 清理答案中的多余内容
            if "Context:" in answer:
                answer = answer.split("Context:")[-1]
            if "Question:" in answer:
                answer = answer.split("Question:")[-1]
            if "Answer" in answer and ":" in answer:
                parts = answer.split("Answer")
                for part in parts[1:]:
                    if ":" in part:
                        potential_answer = part.split(":", 1)[1].strip()
                        if potential_answer:
                            answer = potential_answer
                            break
            
            # 移除常见的回答前缀
            prefixes_to_remove = [
                "The answer is", "The answer:", "Answer:", "It is", "This is", 
                "Based on the context", "According to", "From the context"
            ]
            for prefix in prefixes_to_remove:
                if answer.lower().startswith(prefix.lower()):
                    answer = answer[len(prefix):].strip()
                    if answer.startswith(":"):
                        answer = answer[1:].strip()
                    break
            
            # 清理答案格式
            answer = answer.split('\n')[0].strip()  # 只取第一行
            answer = answer.replace('\\n', ' ').replace('\n', ' ')
            answer = ' '.join(answer.split())  # 标准化空格
            
            # 移除末尾的标点符号（除了必要的）
            answer = answer.rstrip('.,;!?')
            
            print(f"Debug - 清理后答案: '{answer}'")
            
            # 验证答案质量 - 如果太长或包含解释性内容，尝试提取核心部分
            words = answer.split()
            if len(words) > 8:  # 如果超过8个词，尝试提取核心
                # 查找专有名词或关键信息
                important_words = []
                for word in words[:8]:  # 只看前8个词
                    if (word[0].isupper() and len(word) > 2) or word.isdigit():
                        important_words.append(word)
                    elif len(important_words) > 0 and not word[0].isupper():
                        # 如果已经有重要词汇，遇到小写词就停止
                        break
                
                if important_words and len(important_words) <= 5:
                    answer = " ".join(important_words)
                else:
                    # 否则取前5个词
                    answer = " ".join(words[:5])
            
            # 最终检查 - 如果答案仍然不合适，使用上下文提取
            if (not answer or 
                len(answer.strip()) < 2 or 
                any(phrase in answer.lower() for phrase in ['based on', 'according to', 'the context', 'explanation'])):
                print("Debug - 答案不合适，从上下文提取")
                return extract_answer_from_context(question, summary)
        
        return answer if answer else "No clear answer"
        
    except Exception as e:
        print(f"Qwen生成答案时出错: {e}")
        import traceback
        traceback.print_exc()
        # 如果出错，尝试从上下文提取答案
        return extract_answer_from_context(question, summary)

def generate_answer_with_qwen_fallback(question, summary, qa_pipeline):
    """
    使用简单提示词的Qwen备用方案
    """
    try:
        # 使用更直接的提示词格式
        prompt = f"""Context: {summary}

Q: {question}
A: (answer in 1-3 words only)"""
        
        response = qa_pipeline(
            prompt,
            max_new_tokens=100,  # 给足够空间，不截断
            num_return_sequences=1,
            temperature=0.1,
            do_sample=True,
            pad_token_id=qa_pipeline.tokenizer.eos_token_id,
            eos_token_id=qa_pipeline.tokenizer.eos_token_id
        )
        
        generated_text = response[0]['generated_text']
        
        # 提取答案部分
        if "A:" in generated_text:
            answer = generated_text.split("A:")[-1].strip()
        else:
            answer = generated_text.replace(prompt, "").strip()
        
        # 处理Qwen特殊格式
        import re
        if "<think>" in answer:
            if "</think>" in answer:
                answer = answer.split("</think>")[-1].strip()
            else:
                # 从think内容中提取简短答案
                think_content = answer.split("<think>")[-1]
                lines = think_content.split('\n')
                for line in lines:
                    line = line.strip()
                    if line and len(line.split()) <= 5:
                        answer = line
                        break
                else:
                    return extract_answer_from_context(question, summary)
        
        # 清理答案
        answer = answer.split('\n')[0].strip()
        answer = answer.replace('(answer in 1-3 words only)', '').strip()
        
        # 移除括号内容
        answer = re.sub(r'\([^)]*\)', '', answer).strip()
        
        words = answer.split()
        if len(words) > 6:
            answer = " ".join(words[:6])
        
        print(f"Debug - 备用方案答案: '{answer}'")
        
        return answer if answer else extract_answer_from_context(question, summary)
        
    except Exception as e:
        print(f"Qwen备用方案失败: {e}")
        return extract_answer_from_context(question, summary)

# 其他函数保持不变...

def extract_answer_from_context(question, summary):
    """
    从上下文中直接提取答案的备用方法
    """
    import re
    question_lower = question.lower()
    
    # 针对具体的问题类型进行匹配
    if "who" in question_lower:
        # 寻找人名 - 优先查找专有名词
        names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', summary)
        # 过滤掉常见的非人名词汇
        filtered_names = [name for name in names if name not in ['Context', 'Question', 'Answer', 'The', 'This', 'That', 'Dutch', 'Moroccan']]
        if filtered_names:
            return filtered_names[0]
    
    elif "when" in question_lower or "what year" in question_lower:
        # 寻找年份
        years = re.findall(r'\b(19|20)\d{2}\b', summary)
        if years:
            return years[0]
    
    elif "where" in question_lower or "city" in question_lower:
        # 寻找地点
        places = re.findall(r'\b[A-Z][a-z]+(?:,\s*[A-Z][a-z]+)*\b', summary)
        # 过滤常见非地名
        filtered_places = [place for place in places if place not in ['Context', 'Question', 'Answer', 'The', 'This', 'That']]
        if filtered_places:
            return filtered_places[0]
    
    elif "what" in question_lower:
        # 对于"what"问题，尝试找到关键词
        words = summary.split()
        if len(words) > 3:
            return " ".join(words[:3])
    
    # 默认返回第一个重要词汇
    words = summary.split()
    important_words = [word for word in words if len(word) > 3 and word[0].isupper()]
    if important_words:
        return important_words[0]
    
    return "Unable to extract answer"

def generate_answer_with_qwen_fallback(question, summary, qa_pipeline):
    """
    使用简单提示词的Qwen备用方案
    """
    try:
        # 使用更简单的提示词，减少截断风险
        prompt = f"{summary[:300]}\n\nQ: {question}\nA:"
        
        response = qa_pipeline(
            prompt,
            max_new_tokens=50,
            num_return_sequences=1,
            temperature=0.1,
            do_sample=True,
            pad_token_id=qa_pipeline.tokenizer.eos_token_id,
            eos_token_id=qa_pipeline.tokenizer.eos_token_id
        )
        
        generated_text = response[0]['generated_text']
        
        # 提取答案部分
        if "\nA:" in generated_text:
            answer = generated_text.split("\nA:")[-1].strip()
        else:
            answer = generated_text.replace(prompt, "").strip()
        
        # 处理Qwen特殊格式
        import re
        answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL)
        answer = answer.replace('<think>', '').replace('</think>', '')
        
        # 清理答案
        answer = answer.split('\n')[0].strip()
        words = answer.split()
        if len(words) > 10:
            answer = " ".join(words[:10])
        
        print(f"Debug - 备用方案答案: '{answer}'")
        
        return answer if answer else extract_answer_from_context(question, summary)
        
    except Exception as e:
        print(f"Qwen备用方案失败: {e}")
        return extract_answer_from_context(question, summary)

def generate_answer(question, summary, qa_pipeline, model_name):
    """
    基于问题和总结生成答案
    """
    if not question or not summary or not qa_pipeline:
        return ""
    
    try:
        # 根据模型类型选择生成方式
        if "qwen" in model_name.lower():
            answer = generate_answer_with_qwen(question, summary, qa_pipeline)
            # 如果主要方式失败，尝试备用方式
            if not answer or answer in ["Unable to generate answer", "No clear answer", "Unable to extract answer"]:
                print("Debug - 尝试Qwen备用方案")
                answer = generate_answer_with_qwen_fallback(question, summary, qa_pipeline)
            return answer
        elif "llama" in model_name.lower():
            return generate_answer_with_llama(question, summary, qa_pipeline)
        elif "distilbert" in model_name.lower() or "bert" in model_name.lower():
            return generate_answer_with_bert(question, summary, qa_pipeline)
        else:
            return generate_answer_traditional(question, summary, qa_pipeline)
            
    except Exception as e:
        print(f"生成答案时出错: {e}")
        # 简单的关键词匹配作为最后备选
        return extract_simple_answer(question, summary)


def generate_answer_with_llama(question, summary, qa_pipeline):
    """
    使用Llama模型生成答案
    """
    try:
        prompt = f"""Context: {summary}

Question: {question}

Based on the context above, provide a brief and accurate answer:"""

        response = qa_pipeline(
            prompt,
            max_new_tokens=30,
            num_return_sequences=1,
            temperature=0.3,
            do_sample=True,
            pad_token_id=qa_pipeline.tokenizer.eos_token_id,
            eos_token_id=qa_pipeline.tokenizer.eos_token_id
        )
        
        generated_text = response[0]['generated_text']
        
        # 提取答案部分
        if "answer:" in generated_text.lower():
            answer = generated_text.lower().split("answer:")[-1].strip()
        else:
            answer = generated_text.split(prompt)[-1].strip()
        
        # 清理答案
        answer = answer.split('\n')[0].strip()
        words = answer.split()
        if len(words) > 10:
            answer = " ".join(words[:10])
        
        return answer if answer else "No clear answer"
        
    except Exception as e:
        print(f"Llama生成答案时出错: {e}")
        return "Unable to generate answer"

def generate_answer_with_bert(question, summary, qa_pipeline):
    """
    使用BERT问答模型生成答案
    """
    try:
        result = qa_pipeline(
            question=question,
            context=summary
        )
        
        # 获取答案和置信度
        answer = result['answer']
        confidence = result.get('score', 0)
        
        # 如果置信度太低，可能没有找到合适的答案
        if confidence < 0.1:
            return "No clear answer found"
        
        return answer.strip()
        
    except Exception as e:
        print(f"BERT生成答案时出错: {e}")
        return "Unable to generate answer"

def generate_answer_traditional(question, summary, qa_pipeline):
    """
    使用传统文本生成模型
    """
    try:
        prompt = f"Context: {summary}\nQuestion: {question}\nAnswer:"
        result = qa_pipeline(
            prompt,
            max_length=len(prompt.split()) + 20,
            num_return_sequences=1,
            temperature=0.5,
            do_sample=True,
            pad_token_id=qa_pipeline.tokenizer.eos_token_id
        )
        answer = result[0]['generated_text'].split("Answer:")[-1].strip()
        answer = answer.split('\n')[0].split('.')[0]
        return answer.strip()
        
    except Exception as e:
        print(f"传统模型生成答案时出错: {e}")
        return "Unable to generate answer"


def extract_simple_answer(question, summary):
    """
    简单的关键词匹配备选方案
    """
    import re
    question_lower = question.lower()
    
    # 寻找一些常见的问题模式
    if "what year" in question_lower or "when" in question_lower:
        # 寻找年份
        years = re.findall(r'\b(19|20)\d{2}\b', summary)
        if years:
            return years[0]
    
    elif "who" in question_lower:
        # 寻找人名
        names = re.findall(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', summary)
        if names:
            return names[0]
    
    elif "where" in question_lower or "city" in question_lower:
        # 寻找地点
        places = re.findall(r'\b[A-Z][a-z]+(?:, [A-Z][a-z]+)?\b', summary)
        if places:
            return places[0]
    
    # 如果没有找到模式，返回总结的前几个词
    words = summary.split()
    return " ".join(words[:3]) if len(words) >= 3 else summary

def process_summary_file(input_file):
    """
    处理包含summary的文件，生成预测答案
    """
    # 加载问答模型
    print("正在加载问答模型...")
    qa_pipeline, model_name = load_qa_model()
    
    if not qa_pipeline:
        print("无法加载任何模型，退出程序")
        return
    
    print(f"使用模型: {model_name}")
    
    # 生成输出文件名
    timestamp = datetime.now().strftime("%Y%m%d")
    input_filename = os.path.basename(input_file)
    output_filename = f"{timestamp}_answer_{input_filename}"
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
                    
                    # 获取question和summary字段
                    question = data.get('question', '')
                    summary = data.get('summary', '')
                    
                    # 生成预测答案
                    print(f"处理进度: {i+1}/{total_lines} - 生成答案中...")
                    pred_answer = generate_answer(question, summary, qa_pipeline, model_name)
                    
                    # 添加pred_answer字段
                    data['pred_answer'] = pred_answer
                    
                    processed_data.append(data)
                    
                    # 显示进度
                    if (i + 1) % 10 == 0:
                        print(f"已处理 {i+1}/{total_lines} 条记录")
                    
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
        
        # 显示一些示例结果
        if processed_data:
            print("\n示例结果:")
            for i, item in enumerate(processed_data[:3]):
                print(f"问题: {item.get('question', '')[:100]}...")
                print(f"真实答案: {item.get('answer', '')}")
                print(f"预测答案: {item.get('pred_answer', '')}")
                print("-" * 50)
        
    except Exception as e:
        print(f"保存文件时出错: {e}")

def main():
    """
    主函数
    """
    input_file = "/home/xhesica/research/outputs/20250605_summary_20250605_extract_20250605_answer_answer_train_limit1000.json"
    process_summary_file(input_file)

if __name__ == "__main__":
    main()