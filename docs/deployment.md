# DPtoolkit Deployment Guide

This guide covers deploying DPtoolkit in various environments.

## Table of Contents

- [Local Development](#local-development)
- [Production Deployment](#production-deployment)
- [Docker Deployment](#docker-deployment)
- [Cloud Deployment](#cloud-deployment)
- [Configuration](#configuration)
- [Security Considerations](#security-considerations)
- [Monitoring & Logging](#monitoring--logging)
- [Troubleshooting](#troubleshooting)

---

## Local Development

### Prerequisites

- Python 3.10 or higher
- pip package manager
- Git

### Setup

```bash
# Clone the repository
git clone https://github.com/fermatrox/DPtoolkit.git
cd DPtoolkit

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # For development

# Install package in development mode
pip install -e .

# Run tests to verify installation
pytest tests/unit -v
```

### Running the Web Interface

```bash
streamlit run frontend/app.py
```

The application will be available at `http://localhost:8501`.

### Development Mode Options

```bash
# Run with auto-reload on file changes
streamlit run frontend/app.py --server.runOnSave true

# Run on a specific port
streamlit run frontend/app.py --server.port 8080

# Run with debug logging
streamlit run frontend/app.py --logger.level debug
```

---

## Production Deployment

### System Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| CPU | 2 cores | 4+ cores |
| RAM | 4 GB | 8+ GB |
| Disk | 2 GB | 10+ GB |
| Python | 3.10 | 3.11+ |

### Installation Steps

1. **Create application user**

```bash
sudo useradd -m -s /bin/bash dptoolkit
sudo su - dptoolkit
```

2. **Clone and setup**

```bash
git clone https://github.com/fermatrox/DPtoolkit.git
cd DPtoolkit

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

3. **Create systemd service**

```bash
sudo nano /etc/systemd/system/dptoolkit.service
```

```ini
[Unit]
Description=DPtoolkit Streamlit Application
After=network.target

[Service]
Type=simple
User=dptoolkit
WorkingDirectory=/home/dptoolkit/DPtoolkit
Environment="PATH=/home/dptoolkit/DPtoolkit/venv/bin"
ExecStart=/home/dptoolkit/DPtoolkit/venv/bin/streamlit run frontend/app.py --server.port 8501 --server.headless true
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

4. **Enable and start service**

```bash
sudo systemctl daemon-reload
sudo systemctl enable dptoolkit
sudo systemctl start dptoolkit
```

5. **Check status**

```bash
sudo systemctl status dptoolkit
```

### Nginx Reverse Proxy

For production deployments, use Nginx as a reverse proxy.

```nginx
# /etc/nginx/sites-available/dptoolkit
server {
    listen 80;
    server_name dptoolkit.example.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 86400;
    }
}
```

Enable with:

```bash
sudo ln -s /etc/nginx/sites-available/dptoolkit /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### HTTPS with Let's Encrypt

```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d dptoolkit.example.com
```

---

## Docker Deployment

### Dockerfile

Create a `Dockerfile` in the project root:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY dp_toolkit/ dp_toolkit/
COPY frontend/ frontend/
COPY setup.py pyproject.toml ./

# Install the package
RUN pip install -e .

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run the application
CMD ["streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
```

### Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  dptoolkit:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./output:/app/output  # For exported files
    environment:
      - DPTOOLKIT_MAX_ROWS=1000000
      - DPTOOLKIT_MAX_MEMORY_GB=4
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 4G
```

### Build and Run

```bash
# Build image
docker-compose build

# Run in foreground
docker-compose up

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

---

## Cloud Deployment

### Streamlit Community Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app"
4. Select your repository
5. Set main file path: `frontend/app.py`
6. Click "Deploy"

### AWS EC2

1. Launch an EC2 instance (t3.medium or larger)
2. Install Python 3.10+
3. Follow the [Production Deployment](#production-deployment) steps
4. Configure Security Group to allow port 8501 (or 80/443 with Nginx)

### Google Cloud Run

```bash
# Build container
gcloud builds submit --tag gcr.io/PROJECT_ID/dptoolkit

# Deploy
gcloud run deploy dptoolkit \
  --image gcr.io/PROJECT_ID/dptoolkit \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2
```

### Azure Container Instances

```bash
# Create resource group
az group create --name dptoolkit-rg --location eastus

# Create container
az container create \
  --resource-group dptoolkit-rg \
  --name dptoolkit \
  --image ghcr.io/fermatrox/dptoolkit:latest \
  --dns-name-label dptoolkit \
  --ports 8501 \
  --cpu 2 \
  --memory 4
```

---

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DPTOOLKIT_MAX_ROWS` | Maximum rows to process | 1,000,000 |
| `DPTOOLKIT_MAX_MEMORY_GB` | Memory limit in GB | 4 |
| `STREAMLIT_SERVER_PORT` | Server port | 8501 |
| `STREAMLIT_SERVER_ADDRESS` | Server address | localhost |

### Streamlit Configuration

Create `.streamlit/config.toml`:

```toml
[server]
port = 8501
address = "0.0.0.0"
headless = true
maxUploadSize = 200  # MB
enableXsrfProtection = true

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
```

---

## Security Considerations

### Data Privacy

- **No persistence**: Data is processed in memory only and cleared on session end
- **No logging of data**: Only metadata (row counts, column names) is logged
- **Session isolation**: Each user session is independent

### Network Security

1. **Use HTTPS**: Always deploy behind HTTPS in production
2. **Firewall**: Restrict access to necessary ports only
3. **VPN/VPC**: Consider deploying within a private network for sensitive use cases

### File Upload Security

- Maximum file size is limited (default 200MB)
- Only specific file formats are accepted (CSV, Excel, Parquet)
- Files are validated before processing

### Recommendations

1. **Don't expose directly**: Use a reverse proxy (Nginx) in production
2. **Authentication**: Add authentication if deploying for multiple users
3. **Audit logging**: Enable logging for compliance requirements
4. **Regular updates**: Keep dependencies updated for security patches

---

## Monitoring & Logging

### Application Logs

Streamlit logs to stdout by default. Capture with:

```bash
streamlit run frontend/app.py 2>&1 | tee -a /var/log/dptoolkit/app.log
```

### Health Check

The application provides a health endpoint at `/_stcore/health`.

```bash
curl http://localhost:8501/_stcore/health
```

### Metrics

For production monitoring, consider adding:

1. **Prometheus metrics**: Add `prometheus-client` for metrics
2. **Application Performance Monitoring**: New Relic, Datadog, etc.
3. **Error tracking**: Sentry for error monitoring

---

## Troubleshooting

### Common Issues

#### Port Already in Use

```bash
# Find process using port
lsof -i :8501

# Kill process
kill -9 <PID>
```

#### Memory Issues

If processing large files causes memory errors:

1. Increase `DPTOOLKIT_MAX_MEMORY_GB`
2. Use Parquet format (more efficient)
3. Process in smaller batches

#### Slow Performance

1. Check system resources: `htop`
2. Verify disk I/O: `iostat`
3. Consider upgrading hardware for production use

#### Permission Errors

```bash
# Fix ownership
sudo chown -R dptoolkit:dptoolkit /home/dptoolkit/DPtoolkit

# Fix permissions
chmod -R 755 /home/dptoolkit/DPtoolkit
```

### Getting Help

- **Issues**: [GitHub Issues](https://github.com/fermatrox/DPtoolkit/issues)
- **Logs**: Check application logs for error details
- **Debug mode**: Run with `--logger.level debug` for verbose output

---

## Updates

### Updating the Application

```bash
cd /home/dptoolkit/DPtoolkit
git pull origin main
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
sudo systemctl restart dptoolkit
```

### Backup

Before updates, backup any configuration:

```bash
cp .streamlit/config.toml config.toml.backup
```

---

## Support Matrix

| Platform | Status | Notes |
|----------|--------|-------|
| Ubuntu 20.04+ | Supported | Recommended for production |
| Debian 11+ | Supported | |
| CentOS 8+ | Supported | |
| Windows Server | Supported | Use Docker for easier deployment |
| macOS | Development only | |
| Docker | Supported | Recommended for portability |
