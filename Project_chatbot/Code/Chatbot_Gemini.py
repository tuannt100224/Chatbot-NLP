# ============================================================
# RAG CHATBOT - TUYỂN SINH ĐẠI HỌC
# Tích hợp: FAISS + HuggingFace Embeddings + Gemini 1.5 Flash
# Fix: Bỏ ViTokenizer (gây dấu _), thêm Gemini normalize typo/teencode
# ============================================================

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
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import json
import contextlib
import pandas as pd
import re
import numpy as np
import logging

# ======================
# IMPORT XỬ LÝ TEENCODE/TYPO
# ======================
from underthesea import text_normalize
from symspellpy import SymSpell, Verbosity
from dotenv import load_dotenv

# ======================
# CONFIG & ENV
# ======================
load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Tắt warnings không cần thiết
logging.getLogger("underthesea").setLevel(logging.ERROR)
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

DEBUG = False  # Đặt True để chạy chế độ debug retriever, False để chạy chatbot

# ======================
# KHỞI TẠO SPELL CHECKER
# Chỉ dùng cho các trường hợp cơ bản.
# Gemini sẽ xử lý typo phức tạp hơn.
# ======================
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)

common_words = [
    "không", "được", "có", "là", "tôi", "bạn", "người", "mình",
    "học", "trường", "sinh viên", "tuyển", "điểm", "năm", "kỳ",
    "ngành", "đại học", "cao học", "hệ", "phương pháp", "cách",
    "học phí", "chi phí", "tiền", "bao nhiêu", "bao nhiêu tiền",
    "cấp chứng chỉ", "bằng cấp", "hộ khẩu", "hành chính",
    "thi thpt", "xét tuyển", "tuyển thẳng", "kiểm tra năng lực",
    "điểm chuẩn", "học bạ", "ngành học", "chuyên ngành",
]

for word in common_words:
    sym_spell.create_dictionary_entry(word, 1)


# ======================
# HÀM NORMALIZE CƠ BẢN (KHÔNG dùng ViTokenizer)
# ViTokenizer đã bị loại bỏ vì nó nối từ bằng dấu "_"
# làm hỏng keyword matching và FAISS search.
# ======================
def normalize_question(text: str) -> str:
    """
    Normalize text tiếng Việt cơ bản:
    1. Lowercase + strip
    2. underthesea text_normalize (xử lý ký tự đặc biệt, khoảng trắng)
    3. Loại bỏ khoảng trắng thừa

    KHÔNG dùng ViTokenizer vì tạo ra dấu gạch dưới (_) làm hỏng pipeline.
    Typo/teencode phức tạp sẽ do gemini_normalize() xử lý riêng.
    """
    # Bước 1: Lowercase
    text = text.lower().strip()

    # Bước 2: Normalize underthesea
    try:
        text = text_normalize(text)
    except Exception as e:
        pass  # Giữ nguyên nếu lỗi

    # Bước 3: Loại bỏ khoảng trắng thừa
    text = " ".join(text.split())

    return text


# ======================
# KHỞI TẠO GEMINI LLM
# Đặt sớm để gemini_normalize() có thể dùng ngay
# ======================
if not GEMINI_API_KEY:
    raise ValueError(
        "❌ Không tìm thấy GEMINI_API_KEY!\n"
        "Tạo file .env với nội dung: GEMINI_API_KEY=your-key-here\n"
        "Lấy API key miễn phí tại: https://aistudio.google.com/app/apikey"
    )

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0.2,         # Thấp = trả lời chính xác, ít sáng tạo
    max_output_tokens=1024,
)

print("✅ Gemini 2.5 Flash đã sẵn sàng")


# ======================
# GEMINI NORMALIZE (dùng AI để sửa typo/teencode)
# Đây là điểm cải tiến chính so với phiên bản cũ.
# ======================
def gemini_normalize(text: str) -> str:
    """
    Dùng Gemini để sửa lỗi chính tả, teencode, viết tắt tiếng Việt.
    Ví dụ:
      "hojc phis 2025"          → "học phí 2025"
      "diem chuan xay dung bn"  → "điểm chuẩn xây dựng bao nhiêu"
      "hoc phi hk1 2025"        → "học phí học kỳ 1 năm 2025"
    """
    try:
        normalize_prompt = (
            "Bạn là công cụ sửa lỗi chính tả tiếng Việt chuyên dụng cho chatbot tuyển sinh.\n"
            "Nhiệm vụ: Chuyển câu sau về dạng tiếng Việt chuẩn, đầy đủ dấu.\n\n"
            "QUY TẮC:\n"
            "- Sửa lỗi chính tả và typo: hojc→học, phis→phí, diem→điểm, chuan→chuẩn\n"
            "- Giải mã teencode/viết tắt phổ biến:\n"
            "    bn / bao nhiu → bao nhiêu\n"
            "    dc / đc       → được\n"
            "    k / ko / khong→ không\n"
            "    hk / hk1 / hk2→ học kỳ 1 / học kỳ 2\n"
            "    mn            → mọi người\n"
            "    ntn           → như thế nào\n"
            "    sv            → sinh viên\n"
            "    ts            → tuyển sinh\n"
            "    ck            → chuyển khoản\n"
            "- Thêm dấu câu tiếng Việt còn thiếu\n"
            "- GIỮ NGUYÊN: số năm (2024, 2025), mã ngành (A00, D01), tên riêng\n"
            "- Chỉ trả về câu đã sửa, KHÔNG giải thích, KHÔNG thêm gì khác\n\n"
            f"Câu gốc: {text}\n"
            "Câu đã sửa:"
        )

        response = llm.invoke([HumanMessage(content=normalize_prompt)])
        result = response.content.strip()

        # Kiểm tra kết quả hợp lệ
        if result and len(result) < len(text) * 4 and len(result) > 0:
            return result

        return text  # Fallback về text gốc nếu kết quả bất thường

    except Exception as e:
        print(f"[Warning] Gemini normalize error: {e}")
        return text


def should_use_gemini_normalize(text: str) -> bool:
    """
    Phát hiện câu hỏi có chứa typo/teencode không.
    Tránh gọi API thừa khi câu hỏi đã chuẩn.

    Logic phát hiện:
    1. Tỉ lệ từ không dấu > 30% (khả năng cao bỏ dấu hoặc typo)
    2. Có viết tắt đặc trưng (bn, hk1, sv, ntn...)
    """
    text_lower = text.lower().strip()
    words = text_lower.split()

    if not words:
        return False

    # Danh sách viết tắt thường gặp → kích hoạt Gemini normalize
    teencode_patterns = {
        "bn", "bao nhiu", "dc", "đc", "ko", "hk1", "hk2", "hk3",
        "sv", "ts", "mn", "ntn", "ck", "hojc", "phis", "diem", "chuan",
    }
    for pattern in teencode_patterns:
        if pattern in text_lower:
            return True

    # Kiểm tra tỉ lệ từ không dấu tiếng Việt
    # Từ chỉ gồm chữ cái Latin thường (không dấu), độ dài > 2
    no_accent_re = re.compile(r"^[a-z]{3,}$")

    # Các từ Latin thường gặp không phải typo (giữ nguyên)
    allowed_latin = {
        "và", "là", "có", "để", "the", "van", "la", "co", "de",
        "nam", "hoc", "phi", "ngan", "diem",  # một số từ dễ nhầm
    }

    no_accent_count = sum(
        1 for w in words
        if no_accent_re.match(w) and w not in allowed_latin
    )

    ratio = no_accent_count / len(words)
    return ratio > 0.3  # Hơn 30% từ không dấu → cần normalize


# ======================
# 1. LOAD DATA
# ======================
print("📂 Đang load dữ liệu...")
documents = []
data_path = "./Data/processed_data"

for file in sorted(os.listdir(data_path)):
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

print(f"✅ Số documents: {len(documents)}")

# ======================
# 2. SPLIT
# ======================
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(documents)
print(f"✅ Số chunks: {len(splits)}")

# ======================
# 3. LOAD INTENT + CATEGORY
# ======================
excel_path = "./Data/raw_data/data_intent_category.xlsx"

intent_df   = pd.read_excel(excel_path, sheet_name="Câu mẫu Intent")
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

print(f"✅ Intents: {len(intent_dict)} | Categories: {len(category_dict)}")

# ======================
# 4. EMBEDDING + FAISS DB
# ======================
print("🔄 Đang khởi tạo embedding model...")
with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

print("🔄 Đang build FAISS index...")
db = FAISS.from_documents(splits, embeddings, normalize_L2=True)
print("✅ FAISS index sẵn sàng")


# ======================
# HELPER FUNCTIONS
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
# Phát hiện năm tường minh trong câu hỏi bằng regex thuần.
# ======================
def extract_year_hint(question: str) -> dict:
    """
    Trích xuất hint năm từ câu hỏi bằng regex thuần.
    - Dạng năm học "2023-2024" → trả về cả range lẫn năm đơn
    - Dạng năm đơn "2023"      → sinh thêm range "2023-2024"
    - Số ngoài 1900-2099       → bỏ qua
    """
    hint = {}

    range_match = re.search(r"((?:19|20)\d{2})[-–]((?:19|20)\d{2})", question)
    if range_match:
        y1, y2 = range_match.group(1), range_match.group(2)
        hint["nam_hoc_range"] = f"{y1}-{y2}"
        hint["nam_don"] = y1
        return hint

    single_match = re.search(r"(?<!\d)((?:19|20)\d{2})(?!\d)", question)
    if single_match:
        y = single_match.group(1)
        hint["nam_don"] = y
        hint["nam_hoc_range"] = f"{y}-{int(y) + 1}"

    return hint


def year_hint_in_doc(content: str, hint: dict) -> bool:
    if hint.get("nam_hoc_range") and hint["nam_hoc_range"] in content:
        return True
    if hint.get("nam_don") and hint["nam_don"] in content:
        return True
    return False


def filter_docs_by_year_hint(docs: list, hint: dict) -> list:
    if not hint:
        return docs
    return [doc for doc in docs if year_hint_in_doc(doc.page_content, hint)]


def extract_ky_hint(question: str):
    """
    Trích xuất kỳ học bằng regex, không qua embedding.
    Nhận dạng: kỳ 1/2/3, HK1/2/3, học kỳ 1/2/3, kì 1/2/3
    """
    match = re.search(
        r"(?:học\s*kỳ|học\s*kì|kỳ|kì|hk)\s*([123])",
        question,
        re.IGNORECASE,
    )
    return match.group(1) if match else None


# ======================
# SLOT EXTRACTION
# ======================
def extract_slots(question: str) -> dict:
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

    year_range  = extracted.pop("nam_hoc", None)
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
        by_range = [doc for doc in result if year_range in doc.page_content] if year_range else []

        if year_single:
            by_single = [doc for doc in result if year_single in doc.page_content]
        else:
            y = year_range.split("-")[0] if year_range else None
            by_single = [doc for doc in result if y in doc.page_content] if y else []

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
def detect_category(question: str) -> str:
    question = normalize_question(question)
    scores = {
        cat: sum(kw.lower() in question for kw in kws)
        for cat, kws in category_dict.items()
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "unknown"


def enhance_query(q: str, slots: dict) -> str:
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


def detect_intent(text: str, threshold: float = 0.70):
    if not _intent_vecs:
        return None
    text = normalize_question(text)
    q_vec = np.array(embeddings.embed_query(text))
    q_vec = q_vec / max(float(np.linalg.norm(q_vec)), 1e-9)
    best_intent, best_score = None, 0.0
    for intent, anchor_mat in _intent_vecs.items():
        score = _cosine_max(anchor_mat, q_vec)
        if score > best_score:
            best_score, best_intent = score, intent
    return best_intent if best_score >= threshold else None


def detect_small_talk(q: str):
    intent = detect_intent(q)
    return INTENT_REPLIES.get(intent) if intent else None


# ======================
# DOMAIN CENTROID (Out-of-domain detection)
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

    question = normalize_question(question)
    q_vec = np.array(embeddings.embed_query(question))
    q_vec = q_vec / max(float(np.linalg.norm(q_vec)), 1e-9)
    score = float(np.dot(_domain_centroid, q_vec))

    if score >= 0.55:
        return "Xin lỗi, tôi chưa có thông tin về nội dung này trong dữ liệu hiện tại."
    else:
        return "Tôi chỉ hỗ trợ các câu hỏi liên quan đến trường học (học phí, điểm chuẩn, tuyển sinh...). Câu hỏi này nằm ngoài phạm vi của tôi."


# ======================
# BUILD TẤT CẢ ANCHOR VECTORS
# ======================
print("🔄 Đang build anchor vectors...")
_build_slot_anchors()
_build_intent_anchors()
_build_domain_centroid()
print("✅ Tất cả anchor vectors đã sẵn sàng\n")


# ======================
# RETRIEVER FUNCTIONS
# ======================
def get_relevant_docs(question: str, threshold: float = 1.0, k: int = 10):
    question = normalize_question(question)
    docs_scores = db.similarity_search_with_score(question, k=k)
    return [doc for doc, score in docs_scores if score < threshold]


def rerank_docs_by_category(docs, category: str):
    if category == "unknown":
        return docs
    keywords = category_dict.get(category, [])
    return sorted(
        docs,
        key=lambda d: sum(kw.lower() in d.page_content.lower() for kw in keywords),
        reverse=True,
    )


def format_docs(inputs: dict) -> str:
    docs     = inputs["docs"]
    q        = inputs["question"]
    slots    = inputs.get("slots", {})
    category = detect_category(q)

    if not docs:
        return f"__NO_RESULT__:{classify_empty_result(q)}"

    # ── YEAR GUARD ──────────────────────────────────────────────────────
    year_hint = extract_year_hint(q)
    if year_hint:
        docs = filter_docs_by_year_hint(docs, year_hint)
        if not docs:
            return f"__NO_RESULT__:{classify_empty_result(q)}"

    # ── KỲ GUARD ────────────────────────────────────────────────────────
    ky_hint = extract_ky_hint(q)
    if ky_hint:
        filtered_ky = [
            doc for doc in docs
            if re.search(rf"(?:kỳ|kì|hk)\s*{ky_hint}", doc.page_content, re.IGNORECASE)
        ]
        if filtered_ky:
            docs = filtered_ky

    # ── CATEGORY FILTER cứng ────────────────────────────────────────────
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

    # ── SLOT FILTER + RERANK ─────────────────────────────────────────────
    docs = filter_docs_by_slots(docs, slots)
    docs = rerank_docs_by_category(docs, category)

    if not docs:
        return f"__NO_RESULT__:{classify_empty_result(q)}"

    return "\n".join(doc.page_content for doc in docs)


# ======================
# PROMPT TEMPLATE
# ======================
template = """
Bạn là trợ lý tư vấn tuyển sinh đại học. Trả lời câu hỏi DỰA TRÊN CONTEXT được cung cấp.

QUY TẮC:
- Chỉ dùng thông tin trong CONTEXT, không bịa đặt
- Nếu CONTEXT không có thông tin → trả lời: "Xin lỗi, tôi chưa có thông tin về câu hỏi này."
- Trả lời ngắn gọn, rõ ràng bằng tiếng Việt
- Nếu có số liệu (điểm, học phí...) hãy trình bày dạng danh sách cho dễ đọc

Chủ đề câu hỏi: {category}

CONTEXT:
{context}

CÂU HỎI: {question}

TRẢ LỜI:
"""
prompt = ChatPromptTemplate.from_template(template)


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


# ============================================================
# HELPER: Xử lý pipeline đầy đủ cho một câu hỏi
# Dùng chung cho cả DEBUG mode và CHATBOT mode
# ============================================================
def process_question(q: str) -> dict:
    """
    Chạy toàn bộ pipeline cho câu hỏi q.
    Trả về dict chứa tất cả thông tin trung gian và kết quả cuối.
    """
    result = {
        "original":    q,
        "normalized":  None,
        "used_gemini": False,
        "enhanced":    None,
        "category":    None,
        "slots":       {},
        "year_hint":   {},
        "ky_hint":     None,
        "docs":        [],
        "context":     None,
        "answer":      None,
        "is_no_result": False,
    }

    # Bước 1: Normalize cơ bản
    q_normalized = normalize_question(q)

    # Bước 2: Gemini normalize nếu phát hiện typo/teencode
    if should_use_gemini_normalize(q):
        q_gemini = gemini_normalize(q_normalized)
        if q_gemini != q_normalized:
            result["used_gemini"] = True
            result["normalized"] = q_gemini
            q_normalized = q_gemini
        else:
            result["normalized"] = q_normalized
    else:
        result["normalized"] = q_normalized

    # Bước 3: Trích xuất thông tin
    slots      = extract_slots(q_normalized)
    year_hint  = extract_year_hint(q_normalized)
    ky_hint    = extract_ky_hint(q_normalized)
    category   = detect_category(q_normalized)
    q_enhanced = enhance_query(q_normalized, slots)

    result.update({
        "enhanced":  q_enhanced,
        "category":  category,
        "slots":     slots,
        "year_hint": year_hint,
        "ky_hint":   ky_hint,
    })

    # Bước 4: Retrieve docs (gốc + enhanced, merge unique)
    docs_orig     = get_relevant_docs(q_normalized, k=10)
    docs_enhanced = get_relevant_docs(q_enhanced, k=10)
    seen = set()
    docs = []
    for d in docs_orig + docs_enhanced:
        key = d.page_content
        if key not in seen:
            seen.add(key)
            docs.append(d)

    # Bước 5: Áp dụng guards
    if year_hint:
        docs = filter_docs_by_year_hint(docs, year_hint)

    if ky_hint:
        filtered_ky = [
            doc for doc in docs
            if re.search(rf"(?:kỳ|kì|hk)\s*{ky_hint}", doc.page_content, re.IGNORECASE)
        ]
        if filtered_ky:
            docs = filtered_ky

    if category != "unknown":
        cat_keywords = category_dict.get(category, [])
        if cat_keywords:
            filtered_cat = [
                doc for doc in docs
                if any(kw.lower() in doc.page_content.lower() for kw in cat_keywords)
            ]
            docs = filtered_cat

    docs = filter_docs_by_slots(docs, slots)
    docs = rerank_docs_by_category(docs, category)
    result["docs"] = docs

    # Bước 6: Format context
    context = format_docs({"docs": docs, "question": q_normalized, "slots": slots})
    result["context"] = context

    if context.startswith("__NO_RESULT__:"):
        result["is_no_result"] = True
        result["answer"] = context.replace("__NO_RESULT__:", "")
        return result

    # Bước 7: Gọi Gemini sinh câu trả lời
    answer = rag_chain.invoke(q_enhanced)
    result["answer"] = answer

    return result


# ======================
# DEBUG MODE
# ======================
if DEBUG:
    print("=" * 60)
    print("🐛 DEBUG MODE - Kiểm tra retriever (không gọi Gemini sinh answer)")
    print("Gõ 'exit' để thoát")
    print("=" * 60)

    while True:
        q = input("\nTest (exit): ").strip()
        if q == "exit":
            break
        if not q:
            continue

        # Kiểm tra small talk trước
        reply = detect_small_talk(q)
        if reply:
            print(f"  Intent : {detect_intent(q)}")
            print(f"  Reply  : {reply}")
            continue

        r = process_question(q)

        print(f"\n  Original      : {r['original']}")
        print(f"  Normalized    : {r['normalized']}")
        print(f"  Gemini used   : {'✅ Có' if r['used_gemini'] else '❌ Không'}")
        print(f"  Enhanced      : {r['enhanced']}")
        print(f"  Category      : {r['category']}")
        print(f"  Slots         : {r['slots']}")
        print(f"  Year hint     : {r['year_hint']}")
        print(f"  Ky hint       : {r['ky_hint']}")

        if r["is_no_result"]:
            print(f"\n  ⛔ No result: {r['answer']}")
            continue

        if not r["docs"]:
            print("  ❌ Không tìm thấy doc nào")
            continue

        for i, d in enumerate(r["docs"]):
            print(f"\n  --- Doc {i+1} ---")
            print(d.page_content)

    exit()


# ======================
# CHATBOT MODE
# ======================
print("=" * 60)
print("🤖 RAG CHATBOT - Tư vấn tuyển sinh đại học")
print("Powered by: FAISS + HuggingFace + Gemini 1.5 Flash")
print("Gõ 'exit' để thoát")
print("=" * 60)

while True:
    q = input("\nBạn hỏi: ").strip()

    if q == "exit":
        print("👋 Tạm biệt!")
        break

    if len(q) < 3:
        print("Chatbot: Xin lỗi, tôi chưa rõ câu hỏi của bạn.")
        continue

    # Kiểm tra small talk
    reply = detect_small_talk(q)
    if reply:
        print(f"Chatbot: {reply}")
        continue

    # Xử lý câu hỏi qua pipeline đầy đủ
    try:
        r = process_question(q)

        # Hiển thị thông tin normalize nếu Gemini đã sửa
        if r["used_gemini"] and r["normalized"] != q.lower():
            print(f"  [Đã hiểu là: \"{r['normalized']}\"]")

        print(f"Chatbot: {r['answer']}")

    except Exception as e:
        print(f"Chatbot: Xin lỗi, đã có lỗi xảy ra. Vui lòng thử lại.\n[Error: {e}]")