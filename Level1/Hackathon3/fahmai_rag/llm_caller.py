# llm_caller.py
from openai import OpenAI
import re

# Typhoon ใช้ OpenAI SDK แต่เปลี่ยน base_url
client = OpenAI(
    api_key="YOUR_TYPHOON_API_KEY",
    base_url="https://api.opentyphoon.ai/v1"
)

def ask_typhoon(question: str, choices: list[str], context_chunks: list[str]) -> int:
    """
    ส่งคำถาม + เอกสาร → รับคำตอบเป็นเลข 1-10
    """
    # รวม chunks เป็น context เดียว
    context = "\n\n---\n\n".join(context_chunks)
    
    # สร้าง choices text
    choices_text = ""
    for i, choice in enumerate(choices, 1):
        choices_text += f"ตัวเลือก {i}: {choice}\n"
    
    prompt = f"""คุณเป็นผู้ช่วยของร้าน FahMai ร้านอิเล็กทรอนิกส์
ตอบโดยใช้เฉพาะข้อมูลใน CONTEXT เท่านั้น ห้ามใช้ความรู้อื่น

=== CONTEXT (ข้อมูลจากร้าน FahMai) ===
{context}

=== คำถาม ===
{question}

=== ตัวเลือก ===
{choices_text}ตัวเลือก 9: ไม่มีข้อมูลนี้ในฐานความรู้ของร้าน FahMai
ตัวเลือก 10: คำถามนี้ไม่เกี่ยวข้องกับร้าน FahMai เลย

=== วิธีตัดสินใจ ===
ขั้น 1: คำถามนี้เกี่ยวกับร้าน FahMai ไหม?
  - ถ้าไม่เกี่ยวเลย (เช่น ถามเรื่องอาหาร ข่าว กีฬา) → ตอบ 10
ขั้น 2: CONTEXT มีคำตอบไหม?
  - ถ้ามี → เลือกตัวเลือก 1-8 ที่ตรงที่สุด
  - ถ้าไม่มี แต่คำถามเกี่ยวกับ FahMai → ตอบ 9

คิดสั้นๆ แล้วตอบด้วยตัวเลขเดียว (1-10):"""

    response = client.chat.completions.create(
        model="typhoon-v2-70b-instruct",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.0  # temperature=0 = ตอบสม่ำเสมอ ไม่สุ่ม
    )
    
    raw_answer = response.choices[0].message.content.strip()
    
    # ดึงตัวเลขออกจากคำตอบ
    numbers = re.findall(r'\b(10|[1-9])\b', raw_answer)
    if numbers:
        return int(numbers[-1])  # เอาตัวเลขสุดท้าย (หลัง reasoning)
    
    return 9  # ถ้า parse ไม่ได้ → ตอบ 9 (ปลอดภัยกว่า)