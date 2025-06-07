import json
import os
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

def load_passages(passage_file, limit=600):
    passages = []
    with open(passage_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            data = json.loads(line.strip())
            passages.append(data['passage'])
    return passages

def load_qa(qa_file):
    qa_list = []
    with open(qa_file, 'r', encoding='utf-8') as f:
        for line in f:
            qa_list.append(json.loads(line.strip()))
    return qa_list

def extract_related_passages(qa_list, passages, model, top_k=5, batch_size=1000):
    """分批处理避免内存问题"""
    print(f"开始向量化 {len(passages)} 个passages...")
    
    # 分批向量化passages
    passage_embs = []
    for i in tqdm(range(0, len(passages), batch_size), desc="Vectorizing passages"):
        batch = passages[i:i+batch_size]
        batch_embs = model.encode(batch, convert_to_tensor=True, show_progress_bar=False)
        passage_embs.append(batch_embs)
    
    # 合并所有向量
    import torch
    passage_embs = torch.cat(passage_embs, dim=0)
    print(f"Passage向量化完成，shape: {passage_embs.shape}")
    
    # 处理问答
    for qa in tqdm(qa_list, desc="Processing QA"):
        query = qa.get('question', '') + ' ' + qa.get('guess', '')
        query_emb = model.encode(query, convert_to_tensor=True)
        
        # 计算相似度
        cos_scores = util.cos_sim(query_emb, passage_embs)[0]
        top_results = cos_scores.topk(top_k)
        related = [passages[idx] for idx in top_results.indices.cpu().numpy()]
        qa['related'] = related
    
    return qa_list

def main(
    passage_file="/home/xhesica/research/data/train_processed/passage_train.json",
    qa_file="/home/xhesica/research/outputs/20250606_answer_answer_train_limit2000.json",
    output_dir="/home/xhesica/research/outputs",
    passage_limit=11000,
    top_k=6
):
    print("加载passage...")
    passages = load_passages(passage_file, passage_limit)
    print(f"共载入{len(passages)}条passage")
    print("加载问答数据...")
    qa_list = load_qa(qa_file)
    print(f"共载入{len(qa_list)}条问答")
    print("加载向量模型...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("提取相关passage...")
    qa_with_related = extract_related_passages(qa_list, passages, model, top_k=top_k)
    # 输出
    date_str = datetime.now().strftime("%Y%m%d")
    input_filename = os.path.basename(qa_file)
    output_file = os.path.join(output_dir, f"{date_str}_extract_{input_filename}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in qa_with_related:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"已保存到: {output_file}")

def diagnose_model_loading():
    """诊断模型加载问题"""
    print("开始诊断...")
    
    # 检查网络连接
    try:
        import requests
        response = requests.get('https://huggingface.co', timeout=10)
        print("✓ 网络连接正常")
    except:
        print("✗ 网络连接有问题")
    
    # 检查内存
    import psutil
    memory = psutil.virtual_memory()
    print(f"✓ 可用内存: {memory.available // (1024**3)}GB")
    
    # 检查缓存目录
    import os
    cache_dir = os.path.expanduser('~/.cache/huggingface')
    if os.path.exists(cache_dir):
        print(f"✓ 缓存目录存在: {cache_dir}")
    else:
        print(f"✗ 缓存目录不存在: {cache_dir}")
    
    # 尝试加载模型
    try:
        print("尝试加载sentence-transformers...")
        from sentence_transformers import SentenceTransformer
        print("✓ sentence-transformers导入成功")
        
        print("尝试加载模型（这可能需要几分钟）...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✓ 模型加载成功")
        
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()