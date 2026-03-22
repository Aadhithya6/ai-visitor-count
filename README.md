# Intelligent Face Tracker with Real-Time Dashboard

A production-grade, full-stack video analytics solution for tracking unique visitors, logging entry/exit events, and visualizing live data through a premium web dashboard.

## 🏆 Hackathon Submission
**This project is a part of a hackathon run by [Katomaran](https://katomaran.com)**

---

## 🏢 Architecture & AI Planning

### 1. Planning the App
The goal was to create a system that can reliably identify unique individuals in a video stream without re-counting them if they leave and return. The strategy involves:
- **Fast Detection**: Using YOLOv8 to locate people/faces in every few frames.
- **Robust Tracking**: Using DeepSORT to maintain identity across frames even if the person turns away briefly.
- **High-Precision Recognition**: Using InsightFace (ArcFace) to generate 512-dimensional embeddings for every new face.
- **Persistent Memory**: Storing embeddings in PostgreSQL to match returning visitors using Cosine Similarity.
- **Real-Time Visibility**: A FastAPI backend to stream the processed video and a modern CSS/JS dashboard for analytics.

### 2. Architecture Diagram
```mermaid
graph TD
    A[Video Source] -->|OpenCV| B[Frame Pre-processing]
    B -->|YOLOv8| C[Detection Box]
    C -->|DeepSORT| D[Track ID]
    D -->|New Track?| E[Face Recognition Pipeline]
    E -->|InsightFace| F[Generate Embedding]
    F -->|SQLAlchemy| G[(PostgreSQL DB)]
    G -->|Match Found?| H[Update Existing Visitor]
    G -->|No Match?| I[Register New Visitor]
    I -->|Log Event| J[Database & File Logs]
    H -->|Log Re-entry| J
    B -->|Stream| K[FastAPI Backend]
    K -->|Multipart Stream| L[Web Dashboard]
    J -->|JSON API| L
```

---

## 🚀 Key Features

- **Live Processing Dashboard**: Premium UI with Glassmorphism design showing the live camera feed with AI overlays.
- **Unique Visitor Counter**: Intelligent counting that distinguishes between new and returning visitors.
- **Real-Time Event Logs**: Persistent record of every person entering and exiting the frame.
- **Multi-Threaded Performance**: Backend runs the heavy AI pipeline in a background thread while the API serves data instantly.
- **Auto-Registration**: System automatically scales as new people are detected, saving high-quality face crops for future recognition.

---

## 📊 Compute Load Estimation

| Component | CPU Load (Approx) | GPU Load (Approx) | Memory (VRAM) |
| :--- | :--- | :--- | :--- |
| **YOLOv8 Detection** | 15-20% (Multi-core) | 10-15% | ~1.2 GB |
| **InsightFace ( buffalo_l )** | 25-30% (Encoding) | 20-25% | ~2.5 GB |
| **DeepSORT Tracking** | 5-10% | < 5% | ~200 MB |
| **FastAPI Backend** | < 5% | N/A | ~150 MB |
| **Total (Overall)** | **40-60% (i7 10th Gen)** | **30-45% (RTX 3060)** | **~4.5 GB** |

---

## ⚙️ Setup Instructions

### 1. Environment Initialization
Ensure you have **Python 3.10+** and a running **PostgreSQL** instance.

```bash
# Activate Virtual Environment
.\.venv\Scripts\activate

# Install Dependencies
pip install -r requirements.txt
```

### 2. Configuration
Update `config.json` with your database credentials:
```json
{
    "video_source": "walking_video.mp4",
    "db_url": "postgresql://postgres:password@localhost:5432/facedb",
    "detection_skip_frames": 3,
    "similarity_threshold": 0.45,
    "log_dir": "logs"
}
```

### 3. Run the Full Application
To start the backend and the face tracking engine simultaneously:
```bash
python api.py
```
Then, open `index.html` in your browser to view the dashboard at `http://localhost:8000`.

---

## 🧪 Sample Output

### Logs
- `[12:01:10] Face 2de97789 ENTERED`
- `[12:05:45] Face 2de97789 EXITED`

### Database Entries
The `events` table stores the `face_id`, `timestamp`, and `image_path` for every detection.
The `faces` table maintains a unique record for every individual.

### Image Logs
High-quality face crops are stored in `logs/entry/` and `logs/exit/` for manual verification.

---

## 📝 Assumptions Made
- The environment has sufficient lighting for face recognition.
- The `buffalo_l` model is used for maximum accuracy (requires ~4GB GPU memory).
- Database tables are auto-created on application startup (Note: current implementation drops tables on init for hackathon isolation).

---

## 🎥 Project Demonstration
**Watch the video explanation here:** [Loom / YouTube Link Placeholder]

---
This project is a part of a hackathon run by https://katomaran.com
