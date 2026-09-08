# 👤 Face Embedding Search

An **embedding-based face recognition app** built with **Streamlit** and **InsightFace (ArcFace model)**.  
Register people by uploading their photo, then identify them by scanning a new image — using **512-dimensional face embeddings** and **cosine similarity**.

---

## ⚙️ How It Works

1. Upload a photo → app generates a **512-dimensional face embedding** using the ArcFace model
2. Store that embedding with the person's name, age, and address
3. Upload a scan photo → app compares embeddings using **cosine similarity**
4. If similarity score ≥ 0.5 → person is identified ✅

---

## 📁 Project Structure

```
face-embedding-search/
├── test_embedding.py    # Main Streamlit app
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variable template
├── .gitignore           # Git ignore rules
└── readme.md            # This file
```

---

## 🚀 Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Geet-Prince/face-embedding-search.git
cd face-embedding-search
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

**Windows:**
```bash
venv\Scripts\activate
```

**Mac/Linux:**
```bash
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Set Up Environment Variables
```bash
cp .env.example .env
```
Edit `.env` and fill in your values:
```env
HF_TOKEN=your_huggingface_token_here
```

---

## ▶️ Run the App

```bash
streamlit run test_embedding.py
```

Opens in your browser at **http://localhost:8501**

---

## 📥 First-Time Model Download

On first run, InsightFace automatically downloads the **`buffalo_l` model (~300 MB)**.  
Make sure you have an internet connection. Subsequent runs use the cached model.

---

## 🖥️ Usage

### ➕ Add Person Tab
1. Upload a clear photo (JPG/PNG) with **only one face**
2. Enter the person's name, age, and address
3. Click **Add Person** → face embedding is generated and stored in session memory

### 🔍 Scan Face Tab
1. Upload a photo to identify
2. Click **Scan**
3. App compares against all registered people and shows the best match + similarity score

---

## ⚠️ Important Notes

- Registered people are stored **in session memory only** — data is lost on app restart
- Each image must contain **exactly one face** — multiple faces will throw an error
- For best accuracy, use a **clear, well-lit, front-facing photo**
- Similarity threshold is set to **0.5** (adjustable in `test_embedding.py`)

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web UI |
| `opencv-python-headless` | Image reading & processing |
| `numpy` | Embedding math & array operations |
| `insightface` | ArcFace face embedding model |
| `onnxruntime` | ONNX backend for InsightFace |
| `scikit-learn` | Cosine similarity computation |
| `python-dotenv` | Load `.env` variables |

---

## ☁️ Deploy on Streamlit Cloud

1. Fork or connect this repo at [share.streamlit.io](https://share.streamlit.io)
2. Set **main file** to `test_embedding.py`
3. Set **Python version** to `3.11` or `3.12` in App Settings
4. Add your secrets under **Settings → Secrets**:
```toml
HF_TOKEN = "your_huggingface_token_here"
```
5. Click **Deploy** 🚀

---

## 📄 License

This project is for educational and learning purposes.
