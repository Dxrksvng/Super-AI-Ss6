# vector_index.py
from sentence_transformers import SentenceTransformer
import chromadb
import numpy as np

# โมเดลนี้รองรับภาษาไทย และรันบน Mac M4 ได้สบาย
MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"

def build_vector_index(chunks: list[dict]):
    """
    สร้าง Vector index ใน ChromaDB
    ChromaDB = database พิเศษที่เก็บและค้นหา vector ได้
    """
    print("กำลังโหลดโมเดล embedding... (ครั้งแรกอาจใช้เวลานาน)")
    model = SentenceTransformer(MODEL_NAME)
    
    # แปลงทุก chunk เป็น vector
    texts = [chunk["text"] for chunk in chunks]
    print(f"กำลัง encode {len(texts)} chunks...")
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)
    
    # สร้าง ChromaDB database
    client = chromadb.PersistentClient(path="./chroma_db")  # บันทึกลง disk
    
    # ลบ collection เก่าถ้ามี (เพื่อ rebuild)
    try:
        client.delete_collection("fahmai")
    except:
        pass
    
    collection = client.create_collection("fahmai")
    
    # เพิ่มข้อมูลเข้า ChromaDB
    collection.add(
        documents=texts,
        embeddings=embeddings.tolist(),
        ids=[str(i) for i in range(len(chunks))],
        metadatas=[{"source": chunk["source"]} for chunk in chunks]
    )
    
    print(f"สร้าง Vector index เสร็จ: {len(chunks)} vectors")
    return model, collection


def vector_search(query: str, model, collection, top_k: int = 20) -> list[int]:
    """
    ค้นหาด้วย Vector similarity
    คืนค่าเป็น list ของ index
    """
    # แปลงคำถามเป็น vector ก่อน
    query_embedding = model.encode([query]).tolist()
    
    # ค้นหา chunks ที่ vector ใกล้เคียงที่สุด
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )
    
    # แปลง id กลับเป็น index
    indices = [int(id) for id in results["ids"][0]]
    return indices