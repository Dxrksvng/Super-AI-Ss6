import pandas as pd
from pathlib import Path
import anthropic
import json

# ใช้ Claude API สร้าง ground truth
client = anthropic.Anthropic(api_key="YOUR_CLAUDE_API_KEY")

def get_claude_answer(question, choices, context):
    choices_txt = "".join(f"ตัวเลือก {i}: {c}\n" for i,c in enumerate(choices,1))
    
    prompt = f"""คุณเป็นผู้เชี่ยวชาญตอบคำถามเกี่ยวกับร้าน FahMai
ใช้ข้อมูลใน CONTEXT ตอบคำถาม

CONTEXT:
{context}

คำถาม: {question}

{choices_txt}ตัวเลือก 9: ไม่มีข้อมูลนี้ในฐานความรู้
ตัวเลือก 10: ไม่เกี่ยวกับร้าน FahMai

ตอบด้วยตัวเลขเดียว (1-10):"""

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=10,
        messages=[{"role": "user", "content": prompt}]
    )
    
    import re
    nums = re.findall(r'\b(10|[1-9])\b', response.content[0].text)
    return int(nums[-1]) if nums else 9

def load_all_docs(kb_path="knowledge_base"):
    docs = {}
    for folder in ["products", "policies", "store_info"]:
        for f in Path(kb_path).joinpath(folder).glob("*.md"):
            docs[f.name] = f.read_text(encoding="utf-8")
    return docs

def main():
    docs = load_all_docs()
    full_context = "\n\n===\n\n".join(
        f"[{name}]\n{text}" for name, text in docs.items()
    )
    
    df = pd.read_csv("questions.csv")
    sub = pd.read_csv("submission.csv")
    
    results = []
    wrong = []
    
    for _, row in df.iterrows():
        qid = row["id"]
        question = row["question"]
        choices = [row[f"choice_{i}"] for i in range(1, 9)]
        
        print(f"Q{qid}: {question[:50]}...")
        claude_ans = get_claude_answer(question, choices, full_context[:50000])
        our_ans = sub[sub["id"]==qid].iloc[0]["answer"]
        
        match = "✅" if claude_ans == our_ans else "❌"
        print(f"  Claude={claude_ans} Ours={our_ans} {match}")
        
        results.append({"id": qid, "claude": claude_ans, "ours": our_ans, "match": claude_ans==our_ans})
        
        if claude_ans != our_ans:
            wrong.append({"id": qid, "question": question, "claude": claude_ans, "ours": our_ans})
    
    results_df = pd.DataFrame(results)
    print(f"\nตรงกัน: {results_df['match'].sum()}/100")
    print(f"\nข้อที่น่าจะผิด:")
    for w in wrong:
        print(f"  Q{w['id']}: Claude={w['claude']} Ours={w['ours']} — {w['question'][:60]}")
    
    pd.DataFrame(results).to_csv("silver_labels.csv", index=False)
    print("\nบันทึก silver_labels.csv เสร็จ!")

if __name__ == "__main__":
    main()