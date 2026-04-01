# chunk_builder.py
import os
import re

def load_all_files(knowledge_base_path: str) -> list[dict]:
    """
    โหลดไฟล์ .md ทุกไฟล์จากโฟลเดอร์
    คืนค่าเป็น list ของ dict ที่มี 'text' และ 'source'
    """
    documents = []
    
    for root, dirs, files in os.walk(knowledge_base_path):
        for filename in files:
            if filename.endswith(".md"):
                filepath = os.path.join(root, filename)
                
                with open(filepath, "r", encoding="utf-8") as f:
                    text = f.read()
                
                documents.append({
                    "text": text,
                    "source": filepath  # เก็บว่ามาจากไฟล์ไหน
                })
    
    print(f"โหลดมาได้ {len(documents)} ไฟล์")
    return documents


def chunk_document(doc: dict) -> list[dict]:
    """
    หั่นเอกสารเป็นชิ้นๆ ตาม ## header ของ Markdown
    ถ้าไม่มี header ก็หั่นตามจำนวนคำ
    """
    text = doc["text"]
    source = doc["source"]
    chunks = []
    
    # ถ้าเป็นไฟล์สินค้า → หั่นตาม ## section
    if "products" in source:
        sections = re.split(r'\n(?=## )', text)
        for section in sections:
            if len(section.strip()) > 30:  # กรองชิ้นที่สั้นเกินไป
                # ดึงชื่อสินค้าจากชื่อไฟล์มาใส่ด้านหน้า
                product_name = os.path.basename(source).replace(".md", "")
                chunk_text = f"[สินค้า: {product_name}]\n{section.strip()}"
                chunks.append({
                    "text": chunk_text,
                    "source": source
                })
    else:
        # ไฟล์ policy/store_info → หั่นแบบ sliding window
        words = text.split()
        window_size = 400   # จำนวนคำต่อชิ้น
        overlap = 80        # คำที่ซ้อนทับกัน (กันข้อมูลหาย)
        
        for i in range(0, len(words), window_size - overlap):
            chunk_text = " ".join(words[i:i + window_size])
            if len(chunk_text) > 50:
                chunks.append({
                    "text": chunk_text,
                    "source": source
                })
    
    return chunks


def build_all_chunks(knowledge_base_path: str) -> list[dict]:
    """ฟังก์ชันรวม: โหลดทุกไฟล์ แล้วหั่นทุกไฟล์"""
    docs = load_all_files(knowledge_base_path)
    all_chunks = []
    
    for doc in docs:
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)
    
    print(f"ได้ chunks ทั้งหมด {len(all_chunks)} ชิ้น")
    return all_chunks