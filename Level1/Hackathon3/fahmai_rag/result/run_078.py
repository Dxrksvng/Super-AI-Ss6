import re, time
import pandas as pd
import numpy as np
from pathlib import Path
from openai import OpenAI
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

TYPHOON_API_KEY = "sk-bFsaN4CPLp4MEoj1Q8ojLWG4jboTYYNPPK4WdawMc6WD54Mg"
MODEL           = "typhoon-v2.5-30b-a3b-instruct"
KB_PATH         = "knowledge_base"
QUESTIONS_PATH  = "questions.csv"
OUTPUT_PATH     = "submission.csv"
EMBED_MODEL     = "paraphrase-multilingual-mpnet-base-v2"
RETRIEVE_N      = 10
FINAL_N         = 3
SLEEP_SEC       = 0.5

client = OpenAI(base_url="https://api.opentyphoon.ai/v1", api_key=TYPHOON_API_KEY)

REWRITE_DICT = {
    "เทิร์น":    "Trade-in แลกเครื่องเก่า",
    "แตะจ่าย":  "NFC Pay ชำระเงิน",
    "ค่าซ่อม":  "Care+ ซ่อมแซม ราคาซ่อม",
    "จอแตก":    "Care+ จอแตก ซ่อม",
    "ยกเลิก":   "ยกเลิกคำสั่งซื้อ cancellation",
    "คืนของ":   "คืนสินค้า return policy",
    "คืนสินค้า": "return policy คืน",
    "ประกัน":   "warranty การรับประกัน",
    "ผ่อน":     "ผ่อนชำระ 0% installment",
    "screen-to-body": "screen to body ratio อัตราส่วนหน้าจอ",
    "SAR":      "SAR ค่าการแผ่รังสี",
    "ส่งของ":   "จัดส่งสินค้า shipping",
    "ส่งเร็ว":  "จัดส่งด่วน express",
    "สมาชิก":   "membership FahMai Points คะแนน",
    "แคร์พลัส": "Care+ FahMai Care+",
    "ว่ายน้ำ":  "กันน้ำ ATM waterproof",
    "คลื่นไฟฟ้าหัวใจ": "ECG",
    "ตัดเสียง":  "ANC Active Noise Cancelling",
    "เสียงรบกวน": "ANC noise cancelling",
    "ผลิตที่ไหน": "ประเทศที่ผลิต country of origin",
}

NOT_FAHMAI = [
    "วันหยุดราชการ", "ดอกเบี้ยเงินฝาก", "ผัดกระเพรา", "สูตรอาหาร",
    "อัตราแลกเปลี่ยน", "ราคาทอง", "หวย", "ฟุตบอล", "การเมือง",
    "พยากรณ์อากาศ", "หุ้น",
]

def rewrite_query(question):
    expanded = question
    for thai, eng in REWRITE_DICT.items():
        if thai in question:
            expanded += f" {eng}"
    return expanded


def load_docs(kb_path):
    docs = []
    for folder in ["products", "policies", "store_info"]:
        for f in sorted(Path(kb_path).joinpath(folder).glob("*.md")):
            text = f.read_text(encoding="utf-8")
            title = text.split("\n")[0].replace("#", "").strip()
            docs.append({"name": f.name, "text": text, "title": title})
    print(f"โหลด {len(docs)} ไฟล์")
    return docs


def build_indexes(docs):
    print("Building BM25...")
    corpus = [(d["title"] + " ") * 3 + d["text"] for d in docs]
    bm25 = BM25Okapi([c.split() for c in corpus])
    print("Loading embedding model...")
    model = SentenceTransformer(EMBED_MODEL)
    print("Encoding docs...")
    texts = [d["title"] + "\n" + d["text"][:600] for d in docs]
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)
    print("Ready!")
    return bm25, model, embeddings


def retrieve(question, docs, bm25, emb_model, corpus_embeddings):
    expanded = rewrite_query(question)

    bm25_scores = bm25.get_scores(expanded.split())
    bm25_top = np.argsort(bm25_scores)[::-1][:RETRIEVE_N].tolist()

    q_emb = emb_model.encode([expanded])
    vec_scores = cosine_similarity(q_emb, corpus_embeddings)[0]
    vec_top = np.argsort(vec_scores)[::-1][:RETRIEVE_N].tolist()

    scores = {}
    for rank, idx in enumerate(bm25_top):
        scores[idx] = scores.get(idx, 0) + 1 / (60 + rank + 1)
    for rank, idx in enumerate(vec_top):
        scores[idx] = scores.get(idx, 0) + 1 / (60 + rank + 1)

    best_ids = sorted(scores, key=scores.get, reverse=True)[:FINAL_N]
    return [docs[i]["text"] for i in best_ids]


def call_api(prompt):
    estimated = len(prompt) // 2
    max_tokens = estimated + 200
    for attempt in range(4):
        try:
            r = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.0
            )
            return r.choices[0].message.content.strip()
        except Exception as e:
            err = str(e)
            match = re.search(r"prompt_tokens: (\d+)", err)
            if match:
                max_tokens = int(match.group(1)) + 200
                continue
            time.sleep(2 ** attempt)
    return ""


def answer(question, choices, context_docs):
    if any(kw in question for kw in NOT_FAHMAI):
        return 10

    context = "\n\n---\n\n".join([d[:3000] for d in context_docs])
    choices_txt = "".join(f"ตัวเลือก {i}: {c}\n" for i, c in enumerate(choices, 1))

    prompt = (
        f"คุณเป็นผู้ช่วยร้าน FahMai (ฟ้าใหม่) ร้านอิเล็กทรอนิกส์\n"
        f"ตอบโดยใช้เฉพาะข้อมูลใน CONTEXT เท่านั้น ห้ามเดา\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"คำถาม: {question}\n\n"
        f"ตัวเลือก:\n{choices_txt}"
        f"ตัวเลือก 9: ไม่มีข้อมูลนี้ในฐานความรู้ของร้าน FahMai\n"
        f"ตัวเลือก 10: คำถามไม่เกี่ยวข้องกับร้าน FahMai เลย\n\n"
        f"กฎ:\n"
        f"- ถ้าคำถามไม่เกี่ยวกับ FahMai เลย ตอบ 10\n"
        f"- ถ้าเกี่ยวกับ FahMai แต่ไม่มีข้อมูลใน CONTEXT ตอบ 9\n"
        f"- ถ้ามีคำตอบใน CONTEXT ตอบ 1-8\n\n"
        f"ตอบด้วยตัวเลขเดียวเท่านั้น (1-10):"
    )

    raw = call_api(prompt)
    nums = re.findall(r"\b(10|[1-9])\b", raw)
    return int(nums[-1]) if nums else 9


def main():
    docs = load_docs(KB_PATH)
    bm25, emb_model, corpus_embeddings = build_indexes(docs)

    df = pd.read_csv(QUESTIONS_PATH)
    print(f"\nเริ่มตอบ {len(df)} คำถาม...\n")

    results = []
    for _, row in df.iterrows():
        qid      = row["id"]
        question = row["question"]
        choices  = [row[f"choice_{i}"] for i in range(1, 9)]

        print(f"Q{qid}: {question[:55]}...")

        context_docs = retrieve(question, docs, bm25, emb_model, corpus_embeddings)
        ans = answer(question, choices, context_docs)
        print(f"  -> {ans}")

        results.append({"id": qid, "answer": ans})
        time.sleep(SLEEP_SEC)

    pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)
    print(f"\nบันทึก {OUTPUT_PATH} เสร็จ!")
    print(pd.read_csv(OUTPUT_PATH)["answer"].value_counts().sort_index())


if __name__ == "__main__":
    main()
