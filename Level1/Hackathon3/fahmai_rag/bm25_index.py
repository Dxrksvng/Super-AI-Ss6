# bm25_index.py
from rank_bm25 import BM25Okapi

def build_bm25(chunks: list[dict]):
    """
    สร้าง BM25 index จาก chunks ทั้งหมด
    BM25 ต้องการ list ของ list คำ (tokenized)
    """
    # Tokenize: แยกแต่ละ chunk เป็น list ของคำ
    # ภาษาไทยไม่มี space แต่ตอนนี้ข้อมูลมี space อยู่แล้ว
    # ถ้า accuracy ไม่พอค่อยเพิ่ม pythainlp ทีหลัง
    tokenized_corpus = [
        chunk["text"].split()
        for chunk in chunks
    ]
    
    bm25 = BM25Okapi(tokenized_corpus)
    print("สร้าง BM25 index เสร็จแล้ว")
    return bm25


def bm25_search(query: str, bm25, top_k: int = 20) -> list[int]:
    """
    ค้นหาด้วย BM25
    คืนค่าเป็น list ของ index (ตำแหน่ง) ที่ดีที่สุด top_k อัน
    """
    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)
    
    # เรียงจากคะแนนสูงสุด แล้วเอา top_k
    import numpy as np
    top_indices = np.argsort(scores)[::-1][:top_k].tolist()
    
    return top_indices  # คืนเป็น index ของ chunks