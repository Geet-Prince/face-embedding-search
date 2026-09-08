# ?? Face Recognition System

A **Streamlit-based face recognition app** that uses **InsightFace (ArcFace model)** to generate face embeddings, register people, and identify them by scanning a photo.

---

## ?? How It Works

1. Upload a photo of a person ? app generates a **512-dimensional face embedding** using the ArcFace model
2. Store that embedding with their name, age, and address
3. Upload a scan photo ? app compares embeddings using **cosine similarity**
4. If similarity score = 0.5 ? person is identified ?

---

## ?? Project Structure

```
3.Embedding model facescan/
+-- test_embedding.py    # Main Streamlit app
+-- requirement.txt      # Dependencies
+-- readme.md            # This file
```

---

## ?? Setup & Installation

### 1. Create a Virtual Environment
```bash
python -m venv venv
```

### 2. Activate the Virtual Environment

**Windows:**
```bash
venv\Scripts\activate
```

**Mac/Linux:**
```bash
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirement.txt
```

### 4. Create a .env File (Optional)
If you use Hugging Face models, add your token:
```env
HF_TOKEN=your_huggingface_token_here
```

---

## ?? Run the App

```bash
streamlit run test_embedding.py
```

The app will open in your browser at **http://localhost:8501**

---

## ?? First-Time Model Download

On first run, InsightFace will **automatically download** the uffalo_l model (~300 MB).  
Make sure you have an internet connection.

---

## ??? Usage

### ? Add Person Tab
1. Upload a clear photo (JPG/PNG) with **only one face**
2. Enter name, age, and address
3. Click **Add Person** ? face is registered in memory

### ?? Scan Face Tab
1. Upload a photo to identify
2. Click **Scan**
3. App shows the matched person and similarity score

---

## ?? Notes

- Registered people are stored **in session memory only** (lost on app restart)
- Image must contain **exactly one face** — multiple faces will throw an error
- For best accuracy, use a **clear, front-facing photo**
- Similarity threshold is set to **0.5** (adjustable in code)

---

## ?? Dependencies

| Package | Purpose |
|---|---|
| streamlit | Web UI |
| opencv-python | Image reading & processing |
| 
umpy | Embedding math & cosine similarity |
| insightface | ArcFace face embedding model |
| onnxruntime | Backend for InsightFace model |
| python-dotenv | Load .env variables |
