# Sentinel Vision: Enterprise Face Monitoring

A professional, industrial-grade video analytics solution featuring an **Enterprise-Level Dashboard** for real-time tracking of unique visitors, facility security events, and live surveillance demographics.

## 🏆 Hackathon Submission
**This project is a part of a hackathon run by [Katomaran](https://katomaran.com)**

---

## 🏢 Architecture & AI Planning

### 1. Planning the App
Our objective was to develop a systems architecture capable of high-fidelity biometric re-identification in real-time. The planning focused on:
- **Biometric Consistency**: Using ArcFace (InsightFace) to ensure individuals aren't double-counted across multiple sessions.
- **Enterprise UI Design**: Shifting from simple visualizations to a professional sidebar-based dashboard for facility managers.
- **Data Integrity**: Implementing a PostgreSQL-backed audit trail for all entry/exit events.
- **Asynchronous Execution**: Decoupling the heavyweight AI inference loop from the sub-second web API responses.

### 2. Architecture Diagram & Assets
We have provided structured sample data in the `sample_output/` folder:
- **`sample_output/images/`**: High-resolution face captures from live tracking.
- **`sample_output/logs/`**: Historical audit logs in plain text.
- **`sample_output/db_dump/`**: SQL relational schema for visitors and events.

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
    K -->|Multipart Stream| L[Professional Web Dashboard]
    J -->|JSON API| L
```

---

## 🚀 Key Features

- **Enterprise Analytics Dashboard**: A clean, industrial-grade side-navigation UI for real-time facility monitoring.
- **Persistent Biometric Memory**: Securely stores and matches facial embeddings across system reboots.
- **Automated Security Audit**: Every visitor event is logged with a timestamp and high-definition face crop for evidence.
- **Optimized Compute Pipeline**: Engineered for NVIDIA GPU acceleration to maintain high FPS during peak traffic.

---

## 📊 Compute Load Estimation

| Component | CPU Load (Approx) | GPU Load (Approx) | Memory (VRAM) |
| :--- | :--- | :--- | :--- |
| **YOLOv8 Detection** | 15-20% (Multi-core) | 10-15% | ~1.2 GB |
| **InsightFace ( buffalo_l )** | 25-30% (Encoding) | 20-25% | ~2.5 GB |
| **DeepSORT Tracking** | 5-10% | < 5% | ~200 MB |
| **FastAPI Backend** | < 5% | N/A | ~150 MB |
| **Total (Overall)** | **40-60% (i7 10th Gen)** | **30-45% (RTX 3060)** | **~4.5 GB** |

### 📹 Live IP Camera Setup (RTSP)
The system is optimized for high-performance RTSP streams. To connect an IP camera:

1.  **Configure RTSP URL**: Update your `config.json` with your camera's RTSP string:
    ```json
    {
        "video_source": "rtsp://username:password@192.168.1.100:554/stream"
    }
    ```
2.  **Optimizations**: The system automatically detects live streams and applies:
    - **RTSP Over UDP**: To reduce latency and prevent blocky artifacts.
    - **OpenCV Buffer Size (1)**: Ensures you are always processing the most recent frame.
    - **Detection Skip Frames**: Recommended value of `2` or `3` for high-resolution 4K streams.

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
