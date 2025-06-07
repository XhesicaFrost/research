import json
import os
from datetime import datetime
from collections import Counter
import re
import string
import torch
from transformers import pipeline, AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, util

def normalize_text(text):
    """
    标准化文本，用于比较
    """
    if not text:
        return ""
    
    # 转换为小写
    text = text.lower()
    
    # 移除标点符号
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # 移除多余空格
    text = ' '.join(text.split())
    
    return text

def load_semantic_models():
    """
    加载用于语义评分的模型
    """
    models = {}
    
    # 1. 句子相似度模型
    try:
        print("加载句子相似度模型...")
        sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        models['sentence_similarity'] = sentence_model
        print("句子相似度模型加载成功")
    except Exception as e:
        print(f"句子相似度模型加载失败: {e}")
        models['sentence_similarity'] = None
    
    # 2. 自然语言推理模型
    try:
        print("加载自然语言推理模型...")
        nli_model = pipeline(
            "text-classification",
            model="microsoft/DialoGPT-medium",
            device=0 if torch.cuda.is_available() else -1
        )
        models['nli'] = nli_model
        print("自然语言推理模型加载成功")
    except Exception as e:
        print(f"自然语言推理模型加载失败: {e}")
        try:
            # 备选模型
            nli_model = pipeline(
                "text-classification", 
                model="cross-encoder/nli-deberta-v3-base",
                device=0 if torch.cuda.is_available() else -1
            )
            models['nli'] = nli_model
            print("备选NLI模型加载成功")
        except Exception as e2:
            print(f"备选NLI模型也加载失败: {e2}")
            models['nli'] = None
    
    # 3. 问答评估模型
    try:
        print("加载问答评估模型...")
        qa_eval_model = pipeline(
            "text-generation",
            model="gpt2",
            device=0 if torch.cuda.is_available() else -1
        )
        models['qa_eval'] = qa_eval_model
        print("问答评估模型加载成功")
    except Exception as e:
        print(f"问答评估模型加载失败: {e}")
        models['qa_eval'] = None
    
    return models

def sentence_similarity_score(answer, pred_answer, model):
    """
    使用句子嵌入模型计算语义相似度
    """
    if not model or not answer or not pred_answer:
        return 0.0
    
    try:
        # 计算句子嵌入
        embeddings = model.encode([answer, pred_answer], convert_to_tensor=True)
        
        # 计算余弦相似度
        similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
        
        # 确保分数在0-1范围内
        return max(0.0, min(1.0, similarity))
        
    except Exception as e:
        print(f"计算句子相似度时出错: {e}")
        return 0.0

def nli_based_score(answer, pred_answer, model):
    """
    使用自然语言推理模型评估答案一致性
    """
    if not model or not answer or not pred_answer:
        return 0.0
    
    try:
        # 构建前提和假设
        premise = f"The correct answer is: {answer}"
        hypothesis = f"The predicted answer is: {pred_answer}"
        
        # 进行推理
        result = model(f"{premise} [SEP] {hypothesis}")
        
        # 根据模型输出计算分数
        if isinstance(result, list) and len(result) > 0:
            # 寻找entailment或相似的标签
            for item in result:
                label = item.get('label', '').lower()
                score = item.get('score', 0)
                
                if 'entail' in label or 'similar' in label:
                    return score
                elif 'contradict' in label or 'different' in label:
                    return 1.0 - score
        
        return 0.5  # 中性分数
        
    except Exception as e:
        print(f"NLI评分时出错: {e}")
        return 0.0

def llm_based_semantic_score(question, answer, pred_answer, model):
    """
    使用大语言模型进行语义评分
    """
    if not model or not question or not answer or not pred_answer:
        return 0.0
    
    try:
        # 构建评估提示
        prompt = f"""
Question: {question}
Correct Answer: {answer}
Predicted Answer: {pred_answer}

Please evaluate if the predicted answer is semantically equivalent to the correct answer on a scale of 0-10.
Consider the following:
- Do they convey the same meaning?
- Are they factually equivalent?
- Would they be considered correct in the context of the question?

Score (0-10):"""

        result = model(
            prompt,
            max_length=len(prompt.split()) + 100,
            num_return_sequences=1,
            temperature=0.3,
            do_sample=True,
            pad_token_id=model.tokenizer.eos_token_id
        )
        
        generated_text = result[0]['generated_text']
        score_text = generated_text.split("Score (0-10):")[-1].strip()
        
        # 提取数字分数
        import re
        score_match = re.search(r'(\d+(?:\.\d+)?)', score_text)
        if score_match:
            score = float(score_match.group(1))
            # 标准化到0-1范围
            return min(1.0, max(0.0, score / 10.0))
        
        return 0.5  # 默认分数
        
    except Exception as e:
        print(f"LLM语义评分时出错: {e}")
        return 0.0

def exact_match_score(answer, pred_answer):
    """
    精确匹配得分
    """
    norm_answer = normalize_text(answer)
    norm_pred = normalize_text(pred_answer)
    
    return 1.0 if norm_answer == norm_pred else 0.0

def partial_match_score(answer, pred_answer):
    """
    部分匹配得分
    """
    norm_answer = normalize_text(answer)
    norm_pred = normalize_text(pred_answer)
    
    if not norm_answer or not norm_pred:
        return 0.0
    
    # 检查预测答案是否包含真实答案
    if norm_answer in norm_pred:
        return 1.0
    
    # 检查真实答案是否包含预测答案
    if norm_pred in norm_answer:
        return 0.8
    
    # 计算词汇重叠度
    answer_words = set(norm_answer.split())
    pred_words = set(norm_pred.split())
    
    if not answer_words:
        return 0.0
    
    overlap = len(answer_words.intersection(pred_words))
    return overlap / len(answer_words)

def semantic_similarity_score(answer, pred_answer):
    """
    语义相似度得分（简化版本）
    """
    norm_answer = normalize_text(answer)
    norm_pred = normalize_text(pred_answer)
    
    # 数字匹配
    answer_numbers = re.findall(r'\d+', norm_answer)
    pred_numbers = re.findall(r'\d+', norm_pred)
    
    if answer_numbers and pred_numbers:
        if set(answer_numbers) == set(pred_numbers):
            return 1.0
        elif len(set(answer_numbers).intersection(set(pred_numbers))) > 0:
            return 0.7
    
    # 年份匹配
    answer_years = re.findall(r'\b(19|20)\d{2}\b', norm_answer)
    pred_years = re.findall(r'\b(19|20)\d{2}\b', norm_pred)
    
    if answer_years and pred_years:
        if set(answer_years) == set(pred_years):
            return 1.0
    
    # 专有名词匹配
    answer_caps = re.findall(r'\b[A-Z][a-z]+\b', answer)
    pred_caps = re.findall(r'\b[A-Z][a-z]+\b', pred_answer)
    
    if answer_caps and pred_caps:
        caps_overlap = len(set(answer_caps).intersection(set(pred_caps)))
        if caps_overlap > 0:
            return caps_overlap / max(len(set(answer_caps)), len(set(pred_caps)))
    
    return 0.0

def calculate_scores(question, answer, pred_answer, semantic_models):
    """
    计算多种得分，包括AI模型语义评分
    """
    # 传统评分
    exact_score = exact_match_score(answer, pred_answer)
    partial_score = partial_match_score(answer, pred_answer)
    semantic_score = semantic_similarity_score(answer, pred_answer)
    
    # AI模型语义评分
    ai_scores = {}
    
    # 句子相似度评分
    if semantic_models.get('sentence_similarity'):
        ai_scores['sentence_similarity'] = sentence_similarity_score(
            answer, pred_answer, semantic_models['sentence_similarity']
        )
    else:
        ai_scores['sentence_similarity'] = 0.0
    
    # NLI评分
    if semantic_models.get('nli'):
        ai_scores['nli_score'] = nli_based_score(
            answer, pred_answer, semantic_models['nli']
        )
    else:
        ai_scores['nli_score'] = 0.0
    
    # LLM语义评分
    if semantic_models.get('qa_eval'):
        ai_scores['llm_semantic'] = llm_based_semantic_score(
            question, answer, pred_answer, semantic_models['qa_eval']
        )
    else:
        ai_scores['llm_semantic'] = 0.0
    
    # 综合AI语义分数
    ai_semantic_scores = [score for score in ai_scores.values() if score > 0]
    ai_semantic_avg = sum(ai_semantic_scores) / len(ai_semantic_scores) if ai_semantic_scores else 0.0
    
    # 最终综合得分 (包含AI评分)
    composite_score = (
        exact_score * 0.15 +
        partial_score * 0.15 +
        semantic_score * 0.15 +
        ai_semantic_avg * 0.55  # AI模型评分权重较高
    )
    
    return {
        'exact_match': exact_score,
        'partial_match': partial_score,
        'semantic_similarity': semantic_score,
        'ai_sentence_similarity': ai_scores['sentence_similarity'],
        'ai_nli_score': ai_scores['nli_score'],
        'ai_llm_semantic': ai_scores['llm_semantic'],
        'ai_semantic_average': ai_semantic_avg,
        'composite_score': composite_score
    }

def categorize_question_type(question):
    """
    根据问题类型分类
    """
    question_lower = question.lower()
    
    if any(word in question_lower for word in ['what year', 'when', 'date']):
        return 'temporal'
    elif any(word in question_lower for word in ['who', 'which person']):
        return 'person'
    elif any(word in question_lower for word in ['where', 'city', 'country', 'location']):
        return 'location'
    elif any(word in question_lower for word in ['what', 'which']):
        return 'factual'
    elif any(word in question_lower for word in ['how many', 'how much', 'number']):
        return 'numerical'
    elif any(word in question_lower for word in ['yes', 'no', 'is', 'are', 'was', 'were']):
        return 'boolean'
    else:
        return 'other'

def judge_answer_quality(scores):
    """
    判断答案质量等级
    """
    composite = scores['composite_score']
    
    if composite >= 0.9:
        return 'excellent'
    elif composite >= 0.7:
        return 'good'
    elif composite >= 0.5:
        return 'fair'
    elif composite >= 0.3:
        return 'poor'
    else:
        return 'very_poor'

def process_judge_file(input_file):
    """
    处理文件并评估答案质量
    """
    # 加载语义评分模型
    print("正在加载语义评分模型...")
    semantic_models = load_semantic_models()
    
    # 生成输出文件名
    timestamp = datetime.now().strftime("%Y%m%d")
    input_filename = os.path.basename(input_file)
    output_filename = f"{timestamp}_judge_{input_filename}"
    output_file = os.path.join("outputs", output_filename)
    
    # 确保输出目录存在
    os.makedirs("outputs", exist_ok=True)
    
    processed_data = []
    all_scores = []
    question_type_stats = {}
    quality_stats = Counter()
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        total_lines = len(lines)
        print(f"开始评估 {total_lines} 条记录...")
        
        for i, line in enumerate(lines):
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    
                    # 获取答案
                    answer = data.get('answer', '')
                    pred_answer = data.get('pred_answer', '')
                    question = data.get('question', '')
                    
                    # 计算得分（包含AI语义评分）
                    scores = calculate_scores(question, answer, pred_answer, semantic_models)
                    
                    # 分类问题类型
                    question_type = categorize_question_type(question)
                    
                    # 判断质量等级
                    quality = judge_answer_quality(scores)
                    
                    # 添加评估字段
                    data['evaluation'] = {
                        'exact_match_score': scores['exact_match'],
                        'partial_match_score': scores['partial_match'],
                        'semantic_similarity_score': scores['semantic_similarity'],
                        'ai_sentence_similarity_score': scores['ai_sentence_similarity'],
                        'ai_nli_score': scores['ai_nli_score'],
                        'ai_llm_semantic_score': scores['ai_llm_semantic'],
                        'ai_semantic_average': scores['ai_semantic_average'],
                        'composite_score': scores['composite_score'],
                        'question_type': question_type,
                        'answer_quality': quality,
                        'is_correct': scores['composite_score'] >= 0.7
                    }
                    
                    processed_data.append(data)
                    all_scores.append(scores['composite_score'])
                    
                    # 统计
                    if question_type not in question_type_stats:
                        question_type_stats[question_type] = []
                    question_type_stats[question_type].append(scores['composite_score'])
                    quality_stats[quality] += 1
                    
                    if (i + 1) % 10 == 0:
                        print(f"处理进度: {i+1}/{total_lines}")
                    
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
        
        print(f"\n评估完成！共处理 {len(processed_data)} 条记录")
        print(f"输出文件: {output_file}")
        
        # 计算并输出总体评测结果
        print_evaluation_summary(all_scores, question_type_stats, quality_stats, len(processed_data))
        
    except Exception as e:
        print(f"保存文件时出错: {e}")

def print_evaluation_summary(all_scores, question_type_stats, quality_stats, total_count):
    """
    输出总体评测结果
    """
    print("\n" + "="*60)
    print("总体评测结果 (包含AI语义评分)")
    print("="*60)
    
    # 整体性能
    avg_score = sum(all_scores) / len(all_scores) if all_scores else 0
    correct_count = sum(1 for score in all_scores if score >= 0.7)
    accuracy = correct_count / total_count if total_count > 0 else 0
    
    print(f"总记录数: {total_count}")
    print(f"平均综合得分: {avg_score:.3f}")
    print(f"准确率 (得分≥0.7): {accuracy:.3f} ({correct_count}/{total_count})")
    
    # 按问题类型分析
    print(f"\n按问题类型分析:")
    print("-" * 40)
    for q_type, scores in question_type_stats.items():
        type_avg = sum(scores) / len(scores) if scores else 0
        type_correct = sum(1 for score in scores if score >= 0.7)
        type_accuracy = type_correct / len(scores) if scores else 0
        print(f"{q_type:15s}: 平均得分 {type_avg:.3f}, 准确率 {type_accuracy:.3f} ({type_correct}/{len(scores)})")
    
    # 按质量等级分析
    print(f"\n按答案质量分析:")
    print("-" * 40)
    for quality, count in quality_stats.most_common():
        percentage = count / total_count * 100 if total_count > 0 else 0
        print(f"{quality:15s}: {count:3d} ({percentage:5.1f}%)")
    
    # 得分分布
    print(f"\n得分分布:")
    print("-" * 40)
    ranges = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]
    for low, high in ranges:
        count = sum(1 for score in all_scores if low <= score < high)
        if low == 0.9:  # 处理最后一个区间
            count = sum(1 for score in all_scores if low <= score <= high)
        percentage = count / total_count * 100 if total_count > 0 else 0
        print(f"{low:.1f}-{high:.1f}: {count:3d} ({percentage:5.1f}%)")

def main():
    """
    主函数
    """
    input_file = "/home/xhesica/research/outputs/20250607_baseline_answer_answer_train_p11000_q2000.json"
    process_judge_file(input_file)

if __name__ == "__main__":
    main()