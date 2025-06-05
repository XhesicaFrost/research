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
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    trust_remote_code=True  # Qwen模型需要这个参数
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

def load_passages(passage_file, passage_limit=None):
    """加载passages，支持数量限制"""
    passages = []
    print(f"正在加载passages从: {passage_file}")
    
    try:
        with open(passage_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                # 如果设置了passage限制且已达到限制，退出循环
                if passage_limit and len(passages) >= passage_limit:
                    break
                    
                try:
                    data = json.loads(line.strip())
                    passage = data.get('passage', '').strip()
                    if passage:
                        passages.append(passage)
                except json.JSONDecodeError as e:
                    print(f"警告: 第{line_num}行JSON解析错误: {e}")
                    continue
                except Exception as e:
                    print(f"警告: 处理第{line_num}行时出错: {e}")
                    continue
        
        if passage_limit:
            print(f"成功加载 {len(passages)} 个passages (限制: {passage_limit})")
        else:
            print(f"成功加载 {len(passages)} 个passages (无限制)")
        return passages
        
    except FileNotFoundError:
        print(f"错误: 文件未找到: {passage_file}")
        return []
    except Exception as e:
        print(f"错误: 加载passages时出错: {e}")
        return []

def select_relevant_passages(question, passages, top_k=6):
    """
    简单的passage选择策略：基于词汇重叠选择最相关的passages
    """
    import re
    
    # 提取问题中的关键词
    question_words = set(re.findall(r'\w+', question.lower()))
    question_words = {word for word in question_words if len(word) > 2}  # 过滤短词
    
    # 计算每个passage的相关性分数
    passage_scores = []
    for i, passage in enumerate(passages):
        passage_words = set(re.findall(r'\w+', passage.lower()))
        
        # 计算词汇重叠度
        overlap = len(question_words & passage_words)
        
        # 考虑passage长度（避免过长的passage）
        length_penalty = min(1.0, 200 / len(passage.split()))
        
        score = overlap * length_penalty
        passage_scores.append((score, i, passage))
    
    # 按分数排序并选择top_k
    passage_scores.sort(reverse=True, key=lambda x: x[0])
    
    # 选择top_k个passages
    selected_passages = []
    for score, idx, passage in passage_scores[:top_k]:
        if score > 0:  # 只选择有相关性的passages
            selected_passages.append(passage)
    
    print(f"从 {len(passages)} 个passages中选择了 {len(selected_passages)} 个相关passages")
    return selected_passages

def generate_answer_with_qwen(question, context, qa_pipeline):
    """
    使用Qwen模型基于多个passages生成答案
    """
    try:
        # 构建Qwen格式的消息
        messages = [
            {
                "role": "user", 
                "content": f"""Based on the following context passages, please answer the question with ONLY the essential information. Your answer should be brief and direct.

Context passages:
{context}

Question: {question}

Important: Give only the direct answer, no explanation, no thinking process, just the core answer:"""
            }
        ]
        
        response = qa_pipeline(
            messages,
            max_new_tokens=200,  # 给足够的空间
            num_return_sequences=1,
            temperature=0.1,  # 低温度获得更确定的答案
            do_sample=True,
            pad_token_id=qa_pipeline.tokenizer.eos_token_id,
            eos_token_id=qa_pipeline.tokenizer.eos_token_id
        )
        
        print(f"Debug - Qwen原始响应: {response}")
        
        # 提取生成的内容
        answer = ""
        if isinstance(response, list) and len(response) > 0:
            generated_text = response[0].get('generated_text', '')
            
            if isinstance(generated_text, list) and len(generated_text) > 1:
                assistant_reply = generated_text[-1].get('content', '')
                answer = assistant_reply.strip()
            else:
                answer = str(generated_text).strip()
        
        # 处理Qwen的回答
        if answer:
            import re
            
            # 处理<think>标记
            if "<think>" in answer:
                if "</think>" in answer:
                    answer = answer.split("</think>")[-1].strip()
                else:
                    # 从think内容中提取答案
                    think_part = answer.split("<think>")[-1]
                    lines = think_part.split('\n')
                    for line in lines:
                        line = line.strip()
                        if line and len(line.split()) <= 10 and not line.startswith(('The', 'Based', 'According', 'From')):
                            answer = line
                            break
                    else:
                        return extract_answer_from_passages(question, context)
            
            # 清理答案
            prefixes_to_remove = [
                "The answer is", "Answer:", "Based on", "According to", 
                "From the context", "Context passages:"
            ]
            for prefix in prefixes_to_remove:
                if answer.lower().startswith(prefix.lower()):
                    answer = answer[len(prefix):].strip()
                    if answer.startswith(":"):
                        answer = answer[1:].strip()
                    break
            
            # 格式清理
            answer = answer.split('\n')[0].strip()
            answer = ' '.join(answer.split())
            answer = answer.rstrip('.,;!?')
            
            # 长度控制
            words = answer.split()
            if len(words) > 8:
                # 尝试提取核心信息
                important_words = []
                for word in words[:8]:
                    if (word[0].isupper() and len(word) > 2) or word.isdigit():
                        important_words.append(word)
                    elif len(important_words) > 0 and not word[0].isupper():
                        break
                
                if important_words and len(important_words) <= 5:
                    answer = " ".join(important_words)
                else:
                    answer = " ".join(words[:5])
            
            print(f"Debug - 清理后答案: '{answer}'")
            
            if answer and len(answer.strip()) > 1:
                return answer
        
        # 如果无法从模型生成中提取答案，使用备用方法
        return extract_answer_from_passages(question, context)
        
    except Exception as e:
        print(f"Qwen生成答案时出错: {e}")
        import traceback
        traceback.print_exc()
        return extract_answer_from_passages(question, context)

def generate_answer_with_bert(question, context, qa_pipeline):
    """
    使用BERT问答模型生成答案
    """
    try:
        # BERT模型处理长文本时可能有限制，截取前512个词
        context_words = context.split()
        if len(context_words) > 500:
            context = " ".join(context_words[:500])
        
        result = qa_pipeline(
            question=question,
            context=context
        )
        
        answer = result['answer']
        confidence = result.get('score', 0)
        
        print(f"Debug - BERT答案: '{answer}', 置信度: {confidence}")
        
        if confidence < 0.1:
            return extract_answer_from_passages(question, context)
        
        return answer.strip()
        
    except Exception as e:
        print(f"BERT生成答案时出错: {e}")
        return extract_answer_from_passages(question, context)

def generate_answer_traditional(question, context, qa_pipeline):
    """
    使用传统文本生成模型
    """
    try:
        # 截取context以避免超长
        context_words = context.split()
        if len(context_words) > 300:
            context = " ".join(context_words[:300])
        
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        
        result = qa_pipeline(
            prompt,
            max_new_tokens=50,
            num_return_sequences=1,
            temperature=0.5,
            do_sample=True,
            pad_token_id=qa_pipeline.tokenizer.eos_token_id
        )
        
        generated_text = result[0]['generated_text']
        
        if "Answer:" in generated_text:
            answer = generated_text.split("Answer:")[-1].strip()
        else:
            answer = generated_text.replace(prompt, "").strip()
        
        answer = answer.split('\n')[0].split('.')[0].strip()
        
        print(f"Debug - 传统模型答案: '{answer}'")
        
        return answer if answer else extract_answer_from_passages(question, context)
        
    except Exception as e:
        print(f"传统模型生成答案时出错: {e}")
        return extract_answer_from_passages(question, context)

def extract_answer_from_passages(question, context):
    """
    从passages中直接提取答案的备用方法
    """
    import re
    question_lower = question.lower()
    
    # 针对不同问题类型进行提取
    if "who" in question_lower:
        # 寻找人名
        names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', context)
        filtered_names = [name for name in names if name not in ['Context', 'Question', 'Answer']]
        if filtered_names:
            return filtered_names[0]
    
    elif "when" in question_lower or "what year" in question_lower:
        # 寻找年份
        years = re.findall(r'\b(19|20)\d{2}\b', context)
        if years:
            return years[0]
    
    elif "where" in question_lower or "city" in question_lower:
        # 寻找地点
        places = re.findall(r'\b[A-Z][a-z]+(?:,\s*[A-Z][a-z]+)*\b', context)
        filtered_places = [place for place in places if place not in ['Context', 'Question', 'Answer']]
        if filtered_places:
            return filtered_places[0]
    
    elif "what" in question_lower:
        # 对于what问题，找关键词
        words = context.split()
        important_words = [word for word in words if len(word) > 3 and word[0].isupper()]
        if important_words:
            return important_words[0]
    
    # 默认返回第一个有意义的词
    words = context.split()
    for word in words:
        if len(word) > 3 and word[0].isupper():
            return word
    
    return "Unable to extract answer"

def generate_baseline_answer(question, passages, qa_pipeline, model_name):
    """
    基于所有passages生成baseline答案
    """
    if not question or not passages or not qa_pipeline:
        return "No data available"
    
    try:
        # 选择最相关的passages
        relevant_passages = select_relevant_passages(question, passages, top_k=10)
        
        if not relevant_passages:
            return "No relevant passages found"
        
        # 组合passages作为context
        context = "\n\n".join(relevant_passages[:5])  # 限制context长度
        
        # 根据模型类型生成答案
        if "qwen" in model_name.lower():
            answer = generate_answer_with_qwen(question, context, qa_pipeline)
        elif "distilbert" in model_name.lower() or "bert" in model_name.lower():
            answer = generate_answer_with_bert(question, context, qa_pipeline)
        else:
            answer = generate_answer_traditional(question, context, qa_pipeline)
        
        return answer
        
    except Exception as e:
        print(f"生成baseline答案时出错: {e}")
        return extract_answer_from_passages(question, " ".join(passages[:3]))

def process_baseline_qa(input_qa_file, input_passage_file, output_dir, 
                       passage_limit=None, question_limit=None):
    """
    处理问答文件，使用所有passages生成baseline答案
    
    Args:
        input_qa_file: 问答输入文件路径
        input_passage_file: passage输入文件路径  
        output_dir: 输出目录
        passage_limit: passage数量限制 (None表示无限制)
        question_limit: 问题数量限制 (None表示无限制)
    """
    # 加载模型
    print("正在加载问答模型...")
    qa_pipeline, model_name = load_qa_model()
    
    if not qa_pipeline:
        print("无法加载任何模型，退出程序")
        return
    
    print(f"使用模型: {model_name}")
    
    # 加载passages (支持数量限制)
    passages = load_passages(input_passage_file, passage_limit=passage_limit)
    if not passages:
        print("无法加载passages，退出程序")
        return
    
    # 生成输出文件名
    timestamp = datetime.now().strftime("%Y%m%d")
    input_filename = os.path.basename(input_qa_file)
    
    # 在文件名中包含限制信息
    suffix_parts = []
    if passage_limit:
        suffix_parts.append(f"p{passage_limit}")
    if question_limit:
        suffix_parts.append(f"q{question_limit}")
    
    if suffix_parts:
        suffix = "_" + "_".join(suffix_parts)
        output_filename = f"{timestamp}_baseline_answer_{input_filename.replace('.json', f'{suffix}.json')}"
    else:
        output_filename = f"{timestamp}_baseline_answer_{input_filename}"
    
    output_file = os.path.join(output_dir, output_filename)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    processed_data = []
    
    try:
        with open(input_qa_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 应用问题数量限制
        if question_limit:
            lines = lines[:question_limit]
            print(f"限制处理前 {question_limit} 个问题（总共 {len(lines)} 个可用）")
        
        total_lines = len(lines)
        print(f"开始处理 {total_lines} 条问答记录...")
        
        for i, line in enumerate(lines):
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    
                    # 获取question
                    question = data.get('question', '')
                    
                    if question:
                        # 生成baseline预测答案
                        print(f"处理进度: {i+1}/{total_lines} - 问题: {question[:50]}...")
                        pred_answer = generate_baseline_answer(question, passages, qa_pipeline, model_name)
                        
                        # 添加pred_answer字段
                        data['pred_answer'] = pred_answer
                        
                        print(f"生成答案: {pred_answer}")
                    else:
                        data['pred_answer'] = "No question provided"
                    
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
        print(f"文件未找到: {input_qa_file}")
        return
    
    # 保存处理后的数据
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in processed_data:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
        
        print(f"处理完成！共处理 {len(processed_data)} 条记录")
        print(f"输出文件: {output_file}")
        
        # 显示配置信息
        print(f"\n配置信息:")
        print(f"- 使用模型: {model_name}")
        print(f"- Passage数量: {len(passages)} (限制: {passage_limit if passage_limit else '无'})")
        print(f"- 问题数量: {len(processed_data)} (限制: {question_limit if question_limit else '无'})")
        
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
    主函数 - 支持自定义passage和问题数量限制
    """
    input_qa_file = "/home/xhesica/research/data/train_processed/answer_train.json"
    input_passage_file = "/home/xhesica/research/data/train_processed/passage_train.json"
    output_dir = "/home/xhesica/research/outputs"
    
    # 可以在这里修改限制参数
    passage_limit = 6000  # 限制加载1000个passages，设为None表示无限制
    question_limit = 1000   # 限制处理50个问题，设为None表示无限制
    
    print(f"配置信息:")
    print(f"- Passage限制: {passage_limit if passage_limit else '无限制'}")
    print(f"- 问题限制: {question_limit if question_limit else '无限制'}")
    print(f"- 输入QA文件: {input_qa_file}")
    print(f"- 输入Passage文件: {input_passage_file}")
    print(f"- 输出目录: {output_dir}")
    print("-" * 50)
    
    process_baseline_qa(
        input_qa_file=input_qa_file,
        input_passage_file=input_passage_file, 
        output_dir=output_dir,
        passage_limit=passage_limit,
        question_limit=question_limit
    )

if __name__ == "__main__":
    main()