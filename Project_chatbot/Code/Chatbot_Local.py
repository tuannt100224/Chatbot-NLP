# ======================
# Import thư viện
# ======================
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
import os
import json
import contextlib
import pandas as pd
import re
import numpy as np

# ======================
# IMPORT XỬ LÝ TEENCODE/TYPO (NEW)
# ======================
from underthesea import text_normalize
from pyvi import ViTokenizer
from symspellpy import SymSpell, Verbosity
import logging

# Tắt warnings từ underthesea
logging.getLogger("underthesea").setLevel(logging.ERROR)

# ======================
# CONFIG
# ======================
DEBUG = True

# ======================
# Tắt warning
# ======================
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# ======================
# KHỞI TẠO SPELL CHECKER (NEW)
# ======================
# Tạo một instance SymSpell đơn giản (không cần load dictionary ngoài)
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)

# Thêm một số từ phổ biến tiếng Việt để tham chiếu
# Có thể mở rộng sau bằng cách load từ file
common_words = [
    "không", "được", "có", "là", "tôi", "bạn", "người", "mình",
    "học", "trường", "sinh viên", "tuyển", "điểm", "năm", "kỳ",
    "ngành", "đại học", "cao học", "hệ", "phương pháp", "cách",
    "học phí", "chi phí", "tiền", "bao nhiêu", "bao nhiêu tiền",
    "cấp chứng chỉ", "bằng cấp", "hộ khẩu", "hành chính",
    "thi thpt", "xét tuyển", "tuyển thẳng", "kiểm tra năng lực"
]

for word in common_words:
    sym_spell.create_dictionary_entry(word, 1)

# ======================
# HÀM NORMALIZE TEENCODE/TYPO (NEW)
# ======================
def normalize_question(text: str) -> str:
    """
    Xử lý tự động: Teencode + Typo + Viết tắt
    
    Quy trình:
    1. Lowercase
    2. Tokenize tiếng Việt (tách từ ghép)
    3. Normalize từ underthesea (xử lý khoảng trắng, ký tự đặc biệt)
    4. Spell check với symspellpy (sửa lỗi chính tả tự động)
    
    Args:
        text: Câu hỏi gốc từ user
    
    Returns:
        Text đã chuẩn hóa
    """
    # Step 1: Lowercase
    text = text.lower().strip()
    
    # Step 2: Tokenize tách từ ghép (tôi không biết → tôi không biết)
    try:
        text = ViTokenizer.tokenize(text)
    except Exception as e:
        print(f"[Warning] ViTokenizer error: {e}")
        pass  # Nếu tokenizer lỗi, vẫn tiếp tục
    
    # Step 3: Normalize từ underthesea (xử lý khoảng trắng, ký tự)
    try:
        text = text_normalize(text)
    except Exception as e:
        print(f"[Warning] text_normalize error: {e}")
        pass
    
    # Step 4: Spell check từ symspellpy
    # Split thành từ, check từng từ, join lại
    words = text.split()
    corrected_words = []
    
    for word in words:
        # Bỏ qua từ có độ dài < 2 và số (năm học 2024, kỳ 1...)
        if len(word) < 2 or word.isdigit():
            corrected_words.append(word)
            continue
        
        # Spell check
        try:
            suggestions = sym_spell.lookup(
                word, 
                Verbosity.CLOSEST,  # Chỉ trả về từ gần nhất
                max_edit_distance=1,  # Cho phép 1 lỗi chính tả
                include_unknown=True
            )
            
            if suggestions and suggestions[0].distance <= 1:
                # Nếu tìm thấy gợi ý gần, dùng gợi ý
                corrected_words.append(suggestions[0].term)
            else:
                # Không, giữ từ gốc
                corrected_words.append(word)
        except Exception as e:
            print(f"[Warning] Spell check error for '{word}': {e}")
            corrected_words.append(word)
    
    text = ' '.join(corrected_words)
    
    # Loại bỏ khoảng trắng thừa
    text = ' '.join(text.split())
    
    return text

# ======================
# 1. LOAD DATA
# ======================
documents = []
data_path = "./Data/processed_data"

for file in os.listdir(data_path):
    if file.endswith(".json"):
        with open(os.path.join(data_path, file), "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    text = item.get("text", "")
                    if text:
                        documents.append(Document(page_content=text))
            elif isinstance(data, dict):
                text = json.dumps(data, ensure_ascii=False)
                documents.append(Document(page_content=text))

print("Số documents:", len(documents))

# ======================
# 2. SPLIT
# ======================
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(documents)

# ======================
# LOAD INTENT + CATEGORY
# ======================
excel_path = "./Data/raw_data/data_intent_category.xlsx"

intent_df  = pd.read_excel(excel_path, sheet_name="Câu mẫu Intent")
category_df = pd.read_excel(excel_path, sheet_name="Từ khóa Category")

intent_dict   = {}
category_dict = {}

current_intent = None
for _, row in intent_df.iterrows():
    intent   = row["Intent"]
    sentence = row["Câu mẫu"]
    if pd.notna(intent):
        current_intent = intent
    if pd.notna(sentence) and current_intent:
        intent_dict.setdefault(current_intent, []).append(sentence)

current_cat = None
for _, row in category_df.iterrows():
    cat     = row["Category"]
    keyword = row["Từ khóa"]
    if pd.notna(cat):
        current_cat = cat
    if pd.notna(keyword) and current_cat:
        category_dict.setdefault(current_cat, []).append(keyword)

# ======================
# EMBEDDING + DB
# ======================
with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

db = FAISS.from_documents(splits, embeddings, normalize_L2=True)

# ======================
# HELPER
# ======================
def _l2_normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.maximum(norms, 1e-9)

def _embed_phrases(phrases):
    vecs = np.array(embeddings.embed_documents(phrases))
    return _l2_normalize(vecs)

def _cosine_max(anchor_mat: np.ndarray, query_vec: np.ndarray) -> float:
    return float((anchor_mat @ query_vec).max())

# ======================
# SLOT SCHEMA
# ======================
SLOT_SCHEMA = {

    "nam_hoc": {
        "anchors": [
            "năm học 2023-2024", "năm học 2024-2025", "năm học nào",
            "niên khoá", "năm học hiện tại",
        ],
        "pattern": r"(\d{4}[-–]\d{4})",
        "doc_filter": lambda content, val: val in content,
        "threshold": 0.60,
    },

    "nam_don": {
        "anchors": [
            "năm 2024", "năm nay", "năm ngoái", "năm trước",
            "năm tuyển sinh", "năm xét tuyển",
        ],
        "pattern": r"(?<!\d)(\d{4})(?!\d)(?![-–]\d{4})",
        "doc_filter": lambda content, val: val in content,
        "threshold": 0.62,
    },

    "ky_hoc": {
        "anchors": [
            "học kỳ 1", "kỳ 1", "kì một", "học kỳ đầu", "kỳ đầu tiên",
            "semester 1", "HK1", "kỳ mùa thu", "học kỳ thứ nhất",
            "học kỳ 2", "kỳ 2", "kì hai", "học kỳ cuối", "kỳ thứ hai",
            "semester 2", "HK2", "kỳ mùa xuân", "học kỳ thứ hai",
            "học kỳ 3", "kỳ hè", "kỳ phụ", "semester 3", "HK3",
        ],
        "pattern": r"(?:kỳ|kì|k[yỳ]|hk|học kỳ)\s*([1-9IVX]+)",
        "doc_filter": lambda content, val: (
            re.search(rf"(?:kỳ|kì|hk)\s*{re.escape(val)}", content, re.IGNORECASE) is not None
        ),
        "threshold": 0.72,
    },

    "he_dao_tao": {
        "anchors": [
            "hệ đại học", "đại học chính quy", "hệ CQ", "bậc đại học",
            "hệ cao học", "thạc sĩ", "sau đại học", "hệ SĐH",
            "hệ liên thông", "liên thông",
        ],
        "pattern": r"(đại học|cao học|thạc sĩ|liên thông|chính quy)",
        "doc_filter": lambda content, val: val.lower() in content.lower(),
        "threshold": 0.73,
    },

    "phuong_thuc": {
        "anchors": [
            "xét tuyển bằng điểm thi THPT", "thi THPT quốc gia",
            "xét học bạ", "điểm học bạ", "xét tuyển học bạ",
            "đánh giá năng lực", "ĐGNL", "bài thi năng lực",
            "xét tuyển thẳng", "tuyển thẳng",
        ],
        "pattern": r"(thi thpt|học bạ|đgnl|đánh giá năng lực|tuyển thẳng)",
        "doc_filter": lambda content, val: val.lower() in content.lower(),
        "threshold": 0.73,
    },

    "to_hop": {
        "anchors": [
            "tổ hợp A00", "tổ hợp A01", "tổ hợp D01", "tổ hợp môn thi",
            "môn xét tuyển", "khối thi",
        ],
        "pattern": r"\b([A-D]\d{2})\b",
        "doc_filter": lambda content, val: val.upper() in content.upper(),
        "threshold": 0.70,
    },

    "nganh": {
        "anchors": [
            "ngành học", "chuyên ngành", "tên ngành", "ngành tuyển sinh",
            "ngành đào tạo", "mã ngành",
        ],
        "pattern": None,
        "doc_filter": None,
        "threshold": 0.68,
    },
}

# ======================
# BUILD SLOT ANCHOR VECTORS
# ======================
_slot_vecs = {}

def _build_slot_anchors():
    for slot, schema in SLOT_SCHEMA.items():
        _slot_vecs[slot] = _embed_phrases(schema["anchors"])
    

# ======================
# YEAR GUARD
# Phát hiện năm tường minh trong câu hỏi bằng regex thuần (không qua embedding).
# Mục đích: chặn cứng trường hợp câu hỏi có ghi rõ năm nhưng slot
# embedding bỏ sót (năm ngoài ngưỡng cosine, hoặc số không phải năm hợp lệ).
# ======================

def extract_year_hint(question: str) -> dict:
    """
    Trích xuất hint năm từ câu hỏi bằng regex thuần.
    Chỉ nhận năm hợp lệ trong khoảng 1900–2099.
    - Ưu tiên dạng năm học "2023-2024" → trả về cả range lẫn năm đơn.
    - Nếu chỉ có năm đơn "2023" → sinh thêm range "2023-2024".
    - Số ngoài khoảng (100, 202000...) → bỏ qua hoàn toàn.
    Trả về {} nếu không tìm thấy năm hợp lệ.
    """
    hint = {}

    # Ưu tiên dạng năm học: 2023-2024 hoặc 2023–2024
    range_match = re.search(r"((?:19|20)\d{2})[-–]((?:19|20)\d{2})", question)
    if range_match:
        y1, y2 = range_match.group(1), range_match.group(2)
        hint["nam_hoc_range"] = f"{y1}-{y2}"
        hint["nam_don"] = y1
        return hint  # đã đủ, không cần tìm thêm

    # Dạng năm đơn hợp lệ: 19xx hoặc 20xx, không nằm trong số dài hơn
    single_match = re.search(r"(?<!\d)((?:19|20)\d{2})(?!\d)", question)
    if single_match:
        y = single_match.group(1)
        hint["nam_don"] = y
        hint["nam_hoc_range"] = f"{y}-{int(y) + 1}"

    return hint


def year_hint_in_doc(content: str, hint: dict) -> bool:
    """Kiểm tra doc có chứa ít nhất một dạng năm từ hint không."""
    if hint.get("nam_hoc_range") and hint["nam_hoc_range"] in content:
        return True
    if hint.get("nam_don") and hint["nam_don"] in content:
        return True
    return False


def filter_docs_by_year_hint(docs: list, hint: dict) -> list:
    """
    Nếu câu hỏi có hint năm hợp lệ → chỉ giữ doc chứa năm đó.
    Nếu không có hint → trả nguyên docs (không filter).
    """
    if not hint:
        return docs
    return [doc for doc in docs if year_hint_in_doc(doc.page_content, hint)]


def extract_ky_hint(question: str):
    """
    Trích xuất kỳ học bằng regex thuần, không qua embedding.
    Nhận dạng: kỳ 1/2/3, HK1/2/3, học kỳ 1/2/3, kì 1/2/3
    Trả về số kỳ dạng string ("1","2","3") hoặc None.
    """
    match = re.search(
        r"(?:học\s*kỳ|học\s*kì|kỳ|kì|hk)\s*([123])",
        question,
        re.IGNORECASE
    )
    return match.group(1) if match else None


# ======================
# SLOT EXTRACTION
# ======================
def extract_slots(question: str) -> dict:
    # NORMALIZE QUESTION TRƯỚC KHI EXTRACT SLOTS (NEW)
    question_normalized = normalize_question(question)
    
    q_vec = np.array(embeddings.embed_query(question_normalized))
    q_vec = q_vec / max(float(np.linalg.norm(q_vec)), 1e-9)

    extracted = {}

    for slot, schema in SLOT_SCHEMA.items():
        threshold = schema.get("threshold", 0.70)
        score = _cosine_max(_slot_vecs[slot], q_vec)
        if score < threshold:
            continue
        pattern = schema.get("pattern")
        if pattern:
            match = re.search(pattern, question_normalized, re.IGNORECASE)
            extracted[slot] = match.group(1) if match else None
        else:
            extracted[slot] = "__detected__"

    year_range = extracted.pop("nam_hoc", None)
    year_single = extracted.pop("nam_don", None)

    if year_range:
        extracted["nam_hoc_range"] = year_range
    elif year_single:
        extracted["nam_don"] = year_single
        extracted["nam_hoc_range"] = f"{year_single}-{int(year_single)+1}"

    return extracted


# ======================
# FILTER DOCS THEO SLOTS
# ======================
def filter_docs_by_slots(docs, slots: dict) -> list:
    if not slots:
        return docs

    result = docs

    year_range  = slots.get("nam_hoc_range")
    year_single = slots.get("nam_don")

    if year_range or year_single:
        if year_range:
            by_range = [doc for doc in result if year_range in doc.page_content]
        else:
            by_range = []

        if year_single:
            by_single = [doc for doc in result if year_single in doc.page_content]
        else:
            if year_range:
                y = year_range.split("-")[0]
                by_single = [doc for doc in result if y in doc.page_content]
            else:
                by_single = []

        both = [doc for doc in by_range if doc in by_single]
        if both:
            result = both
        elif by_range and by_single:
            result = by_range if len(by_range) <= len(by_single) else by_single
        elif by_range:
            result = by_range
        elif by_single:
            result = by_single
        else:
            return []

    skip_slots = {"nam_hoc_range", "nam_don"}
    for slot, value in slots.items():
        if slot in skip_slots:
            continue
        if value is None or value == "__detected__":
            continue
        doc_filter = SLOT_SCHEMA.get(slot, {}).get("doc_filter")
        if doc_filter is None:
            continue
        filtered = [doc for doc in result if doc_filter(doc.page_content, value)]
        if filtered:
            result = filtered

    return result


# ======================
# CATEGORY DETECTION
# ======================
def normalize_text(text):
    # DÙNG HÀM NORMALIZE_QUESTION THẢ VỀ
    text = normalize_question(text)
    return text

def detect_category(question):
    question = normalize_text(question)
    scores = {cat: sum(kw.lower() in question for kw in kws)
              for cat, kws in category_dict.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "unknown"

def enhance_query(q, slots: dict) -> str:
    # NORMALIZE QUERY TRƯỚC KHI ENHANCE (NEW)
    q = normalize_question(q)
    
    category = detect_category(q)

    if category == "hoc_phi":
        q += " học phí tiền học chi phí"
    elif category == "diem_chuan":
        q += " điểm chuẩn điểm xét tuyển điểm trúng tuyển"

    if slots.get("nam_hoc_range"):
        q += f" năm học {slots['nam_hoc_range']}"
    if slots.get("nam_don"):
        q += f" năm {slots['nam_don']}"

    if slots.get("ky_hoc"):
        q += f" học kỳ {slots['ky_hoc']}"

    if slots.get("phuong_thuc"):
        q += f" {slots['phuong_thuc']}"

    return q


# ======================
# INTENT CLASSIFIER
# ======================
# Key phải khớp CHÍNH XÁC với tên intent trong Excel (đã lowercase + strip).
# Các intent hỏi thông tin (hoi_*) để RAG xử lý → không có trong dict này.
# Chỉ những intent cần trả lời ngay (small talk) mới cần có ở đây.
INTENT_REPLIES = {
    "chao_hoi":          "Xin chào! Tôi là trợ lý tư vấn của trường. Bạn muốn hỏi về điểm chuẩn, học phí hay thông tin tuyển sinh?",
    "tam_biet":          "Tạm biệt! Chúc bạn một ngày tốt lành 👋",
    "cam_on":            "Không có gì! Nếu cần thêm thông tin, bạn cứ hỏi nhé 😊",
    "fallback_khong_ro": "Bạn có thể nói rõ hơn được không? Tôi có thể hỗ trợ về điểm chuẩn, học phí, ngành học, học bổng, hồ sơ và lịch tuyển sinh.",
}

_intent_vecs = {}

def _build_intent_anchors():
    for intent, sentences in intent_dict.items():
        key = intent.strip().lower()
        if key not in INTENT_REPLIES:
            continue
        _intent_vecs[key] = _embed_phrases(sentences)

def detect_intent(text, threshold=0.70):
    if not _intent_vecs:
        return None
    # NORMALIZE TEXT TRƯỚC KHI DETECT (NEW)
    text = normalize_question(text)
    
    q_vec = np.array(embeddings.embed_query(text))
    q_vec = q_vec / max(float(np.linalg.norm(q_vec)), 1e-9)
    best_intent, best_score = None, 0.0
    for intent, anchor_mat in _intent_vecs.items():
        score = _cosine_max(anchor_mat, q_vec)
        if score > best_score:
            best_score, best_intent = score, intent
    return best_intent if best_score >= threshold else None

def detect_small_talk(q):
    intent = detect_intent(q)
    return INTENT_REPLIES.get(intent) if intent else None


# ======================
# DOMAIN CENTROID
# ======================
_domain_centroid = None

def _build_domain_centroid():
    global _domain_centroid
    sample = splits[:200]
    texts = [doc.page_content for doc in sample]
    vecs = np.array(embeddings.embed_documents(texts))
    centroid = vecs.mean(axis=0)
    norm = np.linalg.norm(centroid)
    _domain_centroid = centroid / max(norm, 1e-9)

def classify_empty_result(question: str) -> str:
    if _domain_centroid is None:
        return "Xin lỗi, tôi chưa có thông tin về nội dung này."

    # NORMALIZE QUESTION (NEW)
    question = normalize_question(question)
    
    q_vec = np.array(embeddings.embed_query(question))
    q_vec = q_vec / max(float(np.linalg.norm(q_vec)), 1e-9)
    score = float(np.dot(_domain_centroid, q_vec))

    if score >= 0.55:
        return "Xin lỗi, tôi chưa có thông tin về nội dung này trong dữ liệu hiện tại."
    else:
        return "Tôi chỉ hỗ trợ các câu hỏi liên quan đến trường học (học phí, điểm chuẩn, tuyển sinh...). Câu hỏi này nằm ngoài phạm vi của tôi."

# ======================
# BUILD ALL ANCHORS
# ======================
_build_slot_anchors()
_build_intent_anchors()
_build_domain_centroid()


# ======================
# RETRIEVER FUNCTIONS
# ======================
def get_relevant_docs(question, threshold=1.0, k=10):
    """
    k=10 thay vì 6 để tăng recall — year/ky guard sẽ filter lại sau.
    Dùng query gốc (không enhanced) khi cần tìm chính xác theo kỳ.
    NORMALIZE QUESTION TRƯỚC (NEW)
    """
    question = normalize_question(question)
    docs_scores = db.similarity_search_with_score(question, k=k)
    return [doc for doc, score in docs_scores if score < threshold]

def rerank_docs_by_category(docs, category):
    if category == "unknown":
        return docs
    keywords = category_dict.get(category, [])
    return sorted(docs, key=lambda d: sum(kw.lower() in d.page_content.lower() for kw in keywords), reverse=True)

def format_docs(inputs):
    docs     = inputs["docs"]
    q        = inputs["question"]
    slots    = inputs.get("slots", {})
    category = detect_category(q)

    if not docs:
        return f"__NO_RESULT__:{classify_empty_result(q)}"

    # ── YEAR GUARD ───────────────────────────────────────────────────────
    year_hint = extract_year_hint(q)
    if year_hint:
        docs = filter_docs_by_year_hint(docs, year_hint)
        if not docs:
            return f"__NO_RESULT__:{classify_empty_result(q)}"

    # ── KỲ GUARD: filter cứng theo kỳ học tường minh trong câu hỏi ─────
    ky_hint = extract_ky_hint(q)
    if ky_hint:
        filtered_ky = [
            doc for doc in docs
            if re.search(rf"(?:kỳ|kì|hk)\s*{ky_hint}", doc.page_content, re.IGNORECASE)
        ]
        if filtered_ky:
            docs = filtered_ky

    # ── CATEGORY FILTER cứng: loại doc sai chủ đề sau khi guard xong ──
    # Ví dụ: hỏi học phí nhưng doc trả về là điểm chuẩn → loại bỏ.
    # Chỉ áp dụng khi category rõ ràng (không phải "unknown").
    # keywords dùng chính category_dict để nhất quán với rerank.
    if category != "unknown":
        cat_keywords = category_dict.get(category, [])
        if cat_keywords:
            filtered_cat = [
                doc for doc in docs
                if any(kw.lower() in doc.page_content.lower() for kw in cat_keywords)
            ]
            docs = filtered_cat

            if not docs:
                return f"__NO_RESULT__:{classify_empty_result(q)}"
    # ───────────────────────────────────────────────────────────────────

    docs = filter_docs_by_slots(docs, slots)
    docs = rerank_docs_by_category(docs, category)

    if not docs:
        return f"__NO_RESULT__:{classify_empty_result(q)}"

    return "\n".join(doc.page_content for doc in docs)


# ======================
# DEBUG RETRIEVER
# ======================
if DEBUG:
    while True:
        q = input("\nTest (exit): ")
        if q == "exit":
            break

        reply = detect_small_talk(q)
        if reply:
            print(f"  Intent : {detect_intent(q)}")
            print(f"  Reply  : {reply}")
            continue

        q_normalized = normalize_question(q)  # NORMALIZE TRƯỚC
        slots     = extract_slots(q)
        year_hint = extract_year_hint(q)
        ky_hint   = extract_ky_hint(q)
        q2        = enhance_query(q, slots)
        cat       = detect_category(q)

        print(f"\n  Original      : {q}")
        print(f"  Normalized    : {q_normalized}")
        print(f"  Enhanced      : {q2}")
        print(f"  Category      : {cat}")
        print(f"  Slots         : {slots}")
        print(f"  Year hint     : {year_hint}")
        print(f"  Ky hint       : {ky_hint}")

        # Lấy docs từ cả query gốc lẫn enhanced, hợp nhất để tăng recall
        docs_orig     = get_relevant_docs(q, k=10)
        docs_enhanced = get_relevant_docs(q2, k=10)
        seen = set()
        docs = []
        for d in docs_orig + docs_enhanced:
            key = d.page_content
            if key not in seen:
                seen.add(key)
                docs.append(d)

        if not docs:
            print("  ❌ Không tìm thấy doc nào từ FAISS")
            continue

        # Áp dụng year guard
        if year_hint:
            docs_after_guard = filter_docs_by_year_hint(docs, year_hint)
            if not docs_after_guard:
                msg = classify_empty_result(q)
                print(f"  ⛔ Year guard chặn: không có doc nào khớp năm {year_hint}")
                print(f"  💬 Fallback: {msg}")
                continue
            docs = docs_after_guard

        # Áp dụng kỳ guard
        if ky_hint:
            filtered_ky = [
                doc for doc in docs
                if re.search(rf"(?:kỳ|kì|hk)\s*{ky_hint}", doc.page_content, re.IGNORECASE)
            ]
            if filtered_ky:
                docs = filtered_ky

        # Áp dụng category filter cứng
        if cat != "unknown":
            cat_keywords = category_dict.get(cat, [])
            if cat_keywords:
                filtered_cat = [
                    doc for doc in docs
                    if any(kw.lower() in doc.page_content.lower() for kw in cat_keywords)
                ]

                docs = filtered_cat

                if not docs:
                    msg = classify_empty_result(q)
                    print(f"  ⛔ Category filter chặn")
                    print(f"  💬 Fallback: {msg}")
                    continue

        docs_filtered = filter_docs_by_slots(docs, slots)
        docs_filtered = rerank_docs_by_category(docs_filtered, cat)

        if not docs_filtered:
            msg = classify_empty_result(q)
            print(f"  ⚠️  Không có doc khớp sau filter slot")
            print(f"  💬 Fallback: {msg}")
            continue

        for i, d in enumerate(docs_filtered):
            print(f"\n  --- Doc {i+1} ---")
            print(d.page_content)

    exit()


# ======================
# PROMPT
# ======================
template = """
Bạn là trợ lý tư vấn đại học. Chỉ trả lời dựa trên CONTEXT bên dưới.
Nếu không tìm thấy thông tin → trả lời: "Xin lỗi, tôi chưa có thông tin về câu hỏi này."
Trả lời ngắn gọn bằng tiếng Việt.

Category: {category}

CONTEXT:
{context}

CÂU HỎI: {question}

TRẢ LỜI:
"""
prompt = ChatPromptTemplate.from_template(template)


# ======================
# DEBUG LLM
# ======================
class DebugLLM:
    def invoke(self, prompt):
        print("\n===== PROMPT =====")
        print(prompt)
        return "DEBUG MODE"

llm = DebugLLM()


# ======================
# RAG CHAIN
# ======================
rag_chain = (
    {
        "docs":     lambda q: get_relevant_docs(q),
        "question": RunnablePassthrough(),
        "slots":    lambda q: extract_slots(q),
    }
    | RunnablePassthrough.assign(
        context=format_docs,
        category=lambda x: detect_category(x["question"]),
    )
    | prompt
    | llm
    | StrOutputParser()
)


# ======================
# RUN CHATBOT
# ======================
while True:
    q = input("\nQuestion: ")
    if q == "exit":
        break

    if len(q.strip()) < 3:
        print("Answer: Xin lỗi, tôi chưa rõ câu hỏi của bạn.")
        continue

    reply = detect_small_talk(q)
    if reply:
        print("Answer:", reply)
        continue

    # NORMALIZE QUESTION TRƯỚC KHI XỬ LÝ (NEW)
    q_normalized = normalize_question(q)
    
    slots     = extract_slots(q)
    year_hint = extract_year_hint(q)
    ky_hint   = extract_ky_hint(q)
    q_enhanced = enhance_query(q, slots)

    # Lấy docs từ cả query gốc lẫn enhanced, hợp nhất để tăng recall
    docs_orig     = get_relevant_docs(q, k=10)
    docs_enhanced = get_relevant_docs(q_enhanced, k=10)
    seen = set()
    docs_check = []
    for d in docs_orig + docs_enhanced:
        key = d.page_content
        if key not in seen:
            seen.add(key)
            docs_check.append(d)

    slots_check = extract_slots(q)

    # Áp dụng year guard trước khi format
    if year_hint:
        docs_check = filter_docs_by_year_hint(docs_check, year_hint)

    # Áp dụng kỳ guard
    if ky_hint:
        filtered_ky = [
            doc for doc in docs_check
            if re.search(rf"(?:kỳ|kì|hk)\s*{ky_hint}", doc.page_content, re.IGNORECASE)
        ]
        if filtered_ky:
            docs_check = filtered_ky

    context_check = format_docs({"docs": docs_check, "question": q, "slots": slots_check})

    if context_check.startswith("__NO_RESULT__:"):
        print("Answer:", context_check.replace("__NO_RESULT__:", ""))
    else:
        answer = rag_chain.invoke(q_enhanced)
        print("Answer:", answer)