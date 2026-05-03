# AI Face Attendance System

A robust, real-time facial recognition attendance system with advanced liveness detection and anti-spoofing mechanisms. This project leverages deep learning models for accurate face matching and temporal analysis to ensure physical presence, preventing spoofing attempts using photos or screens.

## 🚀 Features

- **High-Accuracy Face Recognition**: Uses FaceNet (InceptionResnetV1) for generating 512-dimensional facial embeddings.
- **Robust Anti-Spoofing (Liveness Detection)**: Employs a two-stage liveness check (Static and Temporal) to verify physical presence.
- **Multi-Angle Enrollment**: Supports capturing front, left, and right profiles to increase recognition accuracy.
- **In-Memory Caching**: Implements a `StudentCache` layer to minimize database queries and ensure high-speed verification.
- **Web-Based Dashboard**: Flask-powered frontend utilizing the webcam for seamless enrollment and attendance marking.
- **Cloud Database**: Integrated with Supabase (PostgreSQL) for secure, scalable data storage.

---

## 🧠 Technical Approach & Architecture

### 1. Face Detection
We use **OpenCV’s Deep Neural Network (DNN)** module with an SSD (Single Shot MultiBox Detector) framework and a pre-trained ResNet model (`res10_300x300_ssd_iter_140000.caffemodel`). This provides fast and highly accurate facial bounding boxes even under varying lighting conditions.

### 2. Face Alignment & Embedding (Recognition)
Once a face is detected, the system generates a unique mathematical representation (embedding):
*   **Face Alignment**: Before generating embeddings, **MediaPipe Face Mesh** locates the eyes and applies an affine transformation to horizontally align the face. This normalizes head tilt and dramatically improves matching consistency.
*   **Embedding Extraction**: The aligned face crop is passed through an **InceptionResnetV1** model (via `facenet_pytorch`), pre-trained on the `vggface2` dataset. This outputs a 512-d embedding vector.
*   **Matching**: The system calculates the **Cosine Similarity** (via L2-normalization and dot product) between the live embedding and known embeddings stored in the database.

### 3. Liveness Detection & Anti-Spoofing
To prevent attacks using high-resolution prints or digital screens, a highly strict, two-stage evaluation is conducted using **MediaPipe Face Mesh**:

#### Stage 1: Static Per-Frame Checks
*   **Z-Depth Check**: Measures the 3D contour of the face. Flat surfaces (screens/photos) are instantly flagged.
*   **Texture Analysis**: Uses Laplacian variance and Canny edge density to detect the low texture variance typical of printed paper or pixelated screens.
*   **Color Distribution**: Analyzes the YCrCb color space. Screens often emit unnatural color distributions compared to real human skin under natural lighting.

#### Stage 2: Temporal (Multi-Frame) Checks
*   **Blink Detection**: Monitors the **Eye Aspect Ratio (EAR)** over a sliding window. A sudden drop and recovery indicate a natural blink.
*   **Micro-Motion Tracking**: Real humans cannot hold perfectly still; they exhibit micro-movements (breathing, slight sway). The system tracks the standard deviation (variance) of key facial landmarks across multiple frames to differentiate a real face from a perfectly static photo.

### 4. Backend & Database
*   **Flask API**: Serves the frontend and provides RESTful endpoints for `/api/register` and `/api/verify`.
*   **Supabase Client**: Connects to a managed PostgreSQL database storing student details (Name, ERP, Course, Year) alongside their 512-d embedding arrays.
*   **Caching Strategy**: To prevent hitting Supabase on every single video frame, embeddings are cached in server memory and refreshed automatically.

---

## 🛠️ Setup & Installation

### Prerequisites
*   Python 3.9+
*   A Supabase Project (URL and API Key)

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/Hardikvats713/Face-Detection-System.git
   cd Face-Detection-System
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies**
   Ensure you have installed all required libraries. (You can generate a requirements.txt if not already present):
   ```bash
   pip install flask opencv-python numpy mediapipe torch facenet-pytorch supabase
   ```

4. **Environment Variables**
   Create a `.env` file in the root directory and add your Supabase credentials:
   ```ini
   SUPABASE_URL=your_supabase_project_url
   SUPABASE_KEY=your_supabase_anon_key
   ```

5. **Run the Application**
   ```bash
   python app.py
   ```
   Open your browser and navigate to `http://localhost:5000`.

---

## 🤝 Contributing
Contributions are welcome! If you'd like to improve the liveness detection, UI, or overall accuracy, feel free to fork the repository and submit a pull request.

## 📄 License
This project is licensed under the MIT License.
