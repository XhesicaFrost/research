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

def extract_related_passages(qa_list, passages, model, top_k=5):
    # 先向量化所有passage
    passage_embs = model.encode(passages, convert_to_tensor=True, show_progress_bar=True)
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
    qa_file="/home/xhesica/research/outputs/20250605_answer_answer_train_limit1000.json",
    output_dir="/home/xhesica/research/outputs",
    passage_limit=6000,
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

if __name__ == "__main__":
    main()