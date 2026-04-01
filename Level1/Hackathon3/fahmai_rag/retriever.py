# retriever.py
import numpy as np

def rrf_fusion(bm25_indices: list[int], vector_indices: list[int], k: int = 60) -> list[int]:
    """
    Reciprocal Rank Fusion
    k=60 คือค่า default ที่งานวิจัยพบว่าดีที่สุด
    
    สูตร: score = 1/(k + rank)
    อันดับ 1 ได้ 1/61, อันดับ 2 ได้ 1/62 ... ลดลงเรื่อยๆ
    """
    scores = {}
    
    for rank, idx in enumerate(bm25_indices):
        scores[idx] = scores.get(idx, 0) + 1 / (k + rank + 1)
    
    for rank, idx in enumerate(vector_indices):
        scores[idx] = scores.get(idx, 0) + 1 / (k + rank + 1)
    
    # เรียงจากคะแนนสูงสุด
    sorted_indices = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return sorted_indices


def retrieve(query: str, bm25, model, collection, chunks, top_k: int = 5) -> list[str]:
    """
    ฟังก์ชันหลัก: รับคำถาม → คืน chunks ที่เกี่ยวข้องที่สุด
    """
    from bm25_index import bm25_search
    from vector_index import vector_search
    
    # ค้นหาทั้งสองวิธี (เอา top 20 ก่อน แล้วค่อยรวม)
    bm25_results = bm25_search(query, bm25, top_k=20)
    vector_results = vector_search(query, model, collection, top_k=20)
    
    # รวมผลด้วย RRF
    fused = rrf_fusion(bm25_results, vector_results)
    
    # เอาแค่ top_k ที่ดีที่สุด
    best_indices = fused[:top_k]
    best_chunks = [chunks[i]["text"] for i in best_indices]
    
    return best_chunks