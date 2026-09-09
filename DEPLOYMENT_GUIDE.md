# 🚀 CadastreVision Deployment Guide (Step-by-Step)

This guide walks you through deploying the **CadastreVision (SkyGen)** AI & WebGIS platform so that your team and SIH evaluators can access the live working prototype over the public internet.

---

## 🧭 Which Deployment Option Should You Choose?

| Scenario | Best Option | Setup Time | Cost |
| :--- | :--- | :--- | :--- |
| **Instant Live Demo for Evaluators** | **Option 1: Cloudflare Tunnel / ngrok** | **30 Seconds** | **100% Free** |
| **Free Always-On Cloud Hosting** | **Option 2: Render.com** | **3 Minutes** | **100% Free** |
| **AI/ML Showcase & Public Space** | **Option 3: Hugging Face Spaces** | **3 Minutes** | **100% Free** |
| **Self-Hosted Cloud Server** | **Option 4: Docker / AWS EC2 / VPS** | **10 Minutes** | Cloud VM Cost |

---

## ⚡ Option 1: Instant Public URL for Evaluators (Zero Setup, 30s)

If you are demonstrating to the SIH judges from your laptop and want a secure, instant **HTTPS** link that works on phones, tablets, and evaluators' screens without paying or setting up servers:

### Step 1: Start the Application Locally
In your project directory terminal, run:
```bash
# Activate your python virtual environment
source .venv/bin/activate

# Launch the FastAPI WebGIS server
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```
Verify it works by opening: `http://localhost:8000`

### Step 2: Create an Instant Public HTTPS Tunnel
Open a **new terminal tab** and run either Cloudflare Tunnel or ngrok:

#### Method A: Using Cloudflare Tunnel (No account or login needed!)
```bash
npx -y cloudflared tunnel --url http://localhost:8000
```
Cloudflare will immediately generate a public HTTPS URL like:
```text
https://random-words-1234.trycloudflare.com
```
Give this link to anyone in the world — it will connect directly to your live WebGIS interface and API!

#### Method B: Using ngrok
```bash
ngrok http 8000
```

---

## 🌐 Option 2: Deploy on Render.com (100% Free Always-On Web Service)

Render connects directly to your GitHub repository ([`https://github.com/KYaswanthReddy/SIH`](https://github.com/KYaswanthReddy/SIH)) and builds automatically whenever you push new commits.

### Step-by-Step Instructions:

1. **Sign in to Render**:
   - Go to [dashboard.render.com](https://dashboard.render.com).
   - Sign in with your GitHub account (`KYaswanthReddy`).

2. **Create a New Web Service**:
   - Click the **"New +"** button in the top bar.
   - Select **"Web Service"**.

3. **Connect Your Repository**:
   - Choose **"Build and deploy from a Git repository"** and click **Next**.
   - Under your connected repositories, locate `SIH` (`KYaswanthReddy/SIH`) and click **"Connect"**.

4. **Configure the Service**:
   - **Name**: `cadastrevision-skygen` (or any name you prefer)
   - **Region**: Choose closest to India (e.g. `Singapore` or `Frankfurt` or `Oregon`)
   - **Branch**: `main`
   - **Runtime**: **Docker** (Render will automatically detect the [`Dockerfile`](file:///Users/kyashwanth/Documents/sih/Dockerfile) we created)
   - **Instance Type**: **Free** ($0/month)

5. **Advanced Settings (Optional)**:
   - Health Check Path: `/api/patches`
   - Port: `8000`

6. **Click "Create Web Service"**:
   - Render will pull your repo, build the Docker container, install GDAL/PyTorch/FastAPI, and deploy.
   - In 2–3 minutes, you will receive your live URL:
     ```text
     https://cadastrevision-skygen.onrender.com
     ```

7. **Available Live Routes**:
   - **Live WebGIS Interface**: `https://cadastrevision-skygen.onrender.com/`
   - **Interactive PPT Presentation**: `https://cadastrevision-skygen.onrender.com/presentation`
   - **API Documentation (Swagger UI)**: `https://cadastrevision-skygen.onrender.com/docs`
   - **Download PPTX File**: `https://cadastrevision-skygen.onrender.com/SIH26012_Idea_Presentation.pptx`

---

## 🤗 Option 3: Deploy on Hugging Face Spaces (Free AI Cloud Hosting)

Hugging Face Spaces is designed specifically for AI models and hackathon presentations.

### Step-by-Step Instructions:

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space).
2. Set:
   - **Space Name**: `CadastreVision-SkyGen`
   - **License**: `MIT` or `Apache 2.0`
   - **SDK**: Select **Docker** -> **Blank**.
   - **Hardware**: **CPU basic · 2 vCPU · 16 GB · Free**
3. Click **"Create Space"**.
4. Push your repository to Hugging Face:
   ```bash
   git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/CadastreVision-SkyGen
   git push hf main
   ```
5. Hugging Face will build the container and provide a permanent URL:
   ```text
   https://huggingface.co/spaces/YOUR_USERNAME/CadastreVision-SkyGen
   ```

---

## 🐳 Option 4: Deploy on Any Cloud VM (AWS EC2 / GCP / DigitalOcean)

If you have an Ubuntu Linux server or cloud virtual machine:

### Step 1: Install Docker & Docker Compose
```bash
sudo apt update
sudo apt install -y docker.io docker-compose-v2
sudo usermod -aG docker $USER
```

### Step 2: Clone the Repository
```bash
git clone https://github.com/KYaswanthReddy/SIH.git
cd SIH
```

### Step 3: Launch with Docker Compose
```bash
docker compose up -d --build
```
Check status:
```bash
docker compose ps
docker compose logs -f
```
The app will be running on `http://YOUR_SERVER_IP:8000`.

### Step 4: Configure Domain & SSL (Nginx + Certbot)
```nginx
server {
    server_name cadastre.yourdomain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```
Obtain free SSL certificate:
```bash
sudo certbot --nginx -d cadastre.yourdomain.com
```

---

## 🧪 Post-Deployment Health Check Checklist

After deploying to any platform, test these four URLs to ensure 100% functionality:

1. **Frontend WebGIS UI**: `GET /`
   - Should display the interactive map with split-screen slider, layer toggles, and drawing tools.
2. **Patches Catalog**: `GET /api/patches`
   - Should return a JSON array of available cadastral benchmark patches.
3. **AI Inference & Vectorization**: `POST /api/vectorize`
   - Should return GeoJSON polygon boundaries with closed area (m²) and topology report.
4. **Slide Deck & Presentation**: `GET /presentation`
   - Should display the official 16:9 widescreen SIH presentation deck.
