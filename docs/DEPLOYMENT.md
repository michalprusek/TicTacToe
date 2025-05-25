# Deployment Guide

Complete guide for deploying and setting up the Robotic TicTacToe application in production environments.

## System Requirements

### Hardware Requirements

#### Minimum Configuration
- **Computer**: x86_64 or ARM64 processor, 8GB RAM, 50GB storage
- **Robot**: uArm Swift Pro with USB connection
- **Camera**: USB camera with 1080p capability (recommended: Logitech C920)
- **USB Ports**: 2 available USB 3.0 ports

#### Recommended Configuration
- **Computer**: Modern multicore processor, 16GB RAM, SSD storage
- **GPU**: NVIDIA GPU with 4GB+ VRAM for accelerated inference
- **Robot**: uArm Swift Pro with latest firmware
- **Camera**: High-quality USB camera with manual focus capability
- **Lighting**: Consistent LED lighting for detection stability

### Software Requirements

#### Operating System Support
- **Primary**: Ubuntu 20.04+ LTS
- **Secondary**: macOS 11+ (Intel/Apple Silicon)  
- **Limited**: Windows 10+ (requires additional setup)

#### Python Environment
```bash
# Python version
Python 3.8 - 3.11 (recommended: 3.9)

# Core dependencies
PyQt5 >= 5.15.0
OpenCV >= 4.5.0
NumPy >= 1.21.0
PyTorch >= 1.9.0
Ultralytics >= 8.0.0
```

## Installation

### 1. Environment Setup

#### Create Virtual Environment
```bash
# Using conda (recommended)
conda create -n tictactoe python=3.9
conda activate tictactoe

# Using venv
python3.9 -m venv tictactoe_env
source tictactoe_env/bin/activate  # Linux/macOS
# tictactoe_env\Scripts\activate   # Windows
```

#### Install System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install -y \
    python3-dev \
    python3-pip \
    cmake \
    build-essential \
    libopencv-dev \
    libusb-1.0-0-dev \
    udev
```

**macOS:**
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install cmake opencv libusb
```

### 2. Application Installation

#### Clone Repository
```bash
git clone https://github.com/username/TicTacToe.git
cd TicTacToe
```

#### Install Python Dependencies
```bash
# Install core dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install -r requirements-dev.txt
```

#### Download Model Weights
```bash
# Create weights directory
mkdir -p weights

# Download pre-trained models (replace with actual URLs)
wget -O weights/best_detection.pt "https://example.com/models/best_detection.pt"
wget -O weights/best_pose.pt "https://example.com/models/best_pose.pt"

# Verify model files
ls -la weights/
```

### 3. Hardware Setup

#### Robot Configuration

**uArm Swift Pro Setup:**
```bash
# Add user to dialout group for serial access (Linux)
sudo usermod -a -G dialout $USER
sudo usermod -a -G tty $USER

# Logout and login for changes to take effect
# OR restart your session

# Test robot connection
python -c "
import pyuarm
arm = pyuarm.SwiftAPI()
arm.connect()
print(f'Connected: {arm.connected}')
arm.disconnect()
"
```

**Camera Setup:**
```bash
# Test camera detection
python -c "
import cv2
cap = cv2.VideoCapture(0)
print(f'Camera opened: {cap.isOpened()}')
ret, frame = cap.read()
if ret:
    print(f'Frame shape: {frame.shape}')
cap.release()
"

# List available cameras (Linux)
ls /dev/video*

# Test multiple cameras
for i in range(3):
    python -c "
import cv2
cap = cv2.VideoCapture($i)
if cap.isOpened():
    print(f'Camera {$i}: Available')
cap.release()
"
```

#### Workspace Calibration

**Physical Setup:**
1. **Robot Positioning**: Place uArm Swift Pro on stable surface
2. **Camera Mounting**: Position camera for clear game board view
3. **Game Board**: Place high-contrast grid in robot workspace
4. **Lighting**: Set up consistent illumination (LED recommended)

**Calibration Process:**
```bash
# Run hand-eye calibration
python -m app.calibration.calibration

# Test calibration accuracy
python debug_coordinates.py
```

## Configuration

### 1. Application Configuration

#### Main Configuration (`app/core/config.py`)
```python
# Production configuration example
@dataclass
class ProductionConfig:
    detector: GameDetectorConfig = field(default_factory=lambda: GameDetectorConfig(
        camera_index=0,
        frame_width=1920,
        frame_height=1080,
        detection_confidence=0.6,  # Higher for production
        pose_confidence=0.6,
        processing_fps=2.0
    ))
    
    arm: ArmConfig = field(default_factory=lambda: ArmConfig(
        port=None,  # Auto-detect
        safe_z=120.0,  # Conservative safety height
        draw_z=25.0,
        drawing_speed=400,  # Precise movement
        movement_speed=800
    ))
    
    game: GameConfig = field(default_factory=lambda: GameConfig(
        difficulty=7,  # Challenging but not perfect
        max_detection_wait_time=15.0,  # Longer timeout
        max_retry_count=5,
        move_cooldown_seconds=3.0
    ))
```

#### Environment Configuration
```bash
# Create .env file for environment-specific settings
cat > .env << EOF
# Camera settings
CAMERA_INDEX=0
CAMERA_RESOLUTION=1920x1080

# Robot settings  
ROBOT_PORT=/dev/ttyUSB0
ROBOT_SAFE_HEIGHT=120

# AI settings
AI_DIFFICULTY=7
DETECTION_CONFIDENCE=0.6

# Logging
LOG_LEVEL=INFO
LOG_FILE=/var/log/tictactoe/app.log
EOF
```

### 2. System Configuration

#### Systemd Service (Linux)
```bash
# Create service file
sudo tee /etc/systemd/system/tictactoe.service > /dev/null << EOF
[Unit]
Description=Robotic TicTacToe Application
After=network.target

[Service]
Type=simple
User=tictactoe
Group=tictactoe
WorkingDirectory=/opt/tictactoe
Environment=DISPLAY=:0
ExecStart=/opt/tictactoe/venv/bin/python -m app.main.main_pyqt
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable tictactoe
sudo systemctl start tictactoe
```

#### udev Rules for Hardware
```bash
# Create udev rules for consistent device naming
sudo tee /etc/udev/rules.d/99-tictactoe.rules > /dev/null << EOF
# uArm Swift Pro
SUBSYSTEM=="tty", ATTRS{idVendor}=="2341", ATTRS{idProduct}=="0042", SYMLINK+="ttyROBOT"

# Camera (adjust vendor/product IDs as needed)
SUBSYSTEM=="video4linux", ATTRS{idVendor}=="046d", ATTRS{idProduct}=="085b", SYMLINK+="videoCHESS"
EOF

# Reload udev rules
sudo udevadm control --reload-rules
sudo udevadm trigger
```

## Docker Deployment

### 1. Docker Configuration

#### Dockerfile
```dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    libopencv-dev \
    libusb-1.0-0-dev \
    libqt5gui5 \
    libqt5widgets5 \
    libqt5core5a \
    x11-apps \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 tictactoe
RUN chown -R tictactoe:tictactoe /app
USER tictactoe

# Expose any necessary ports
EXPOSE 8080

# Set environment variables
ENV PYTHONPATH=/app
ENV QT_X11_NO_MITSHM=1

# Default command
CMD ["python", "-m", "app.main.main_pyqt"]
```

#### Docker Compose
```yaml
version: '3.8'

services:
  tictactoe:
    build: .
    container_name: tictactoe-app
    environment:
      - DISPLAY=${DISPLAY}
      - QT_X11_NO_MITSHM=1
    volumes:
      - /tmp/.X11-unix:/tmp/.X11-unix:rw
      - ./config:/app/config
      - ./logs:/app/logs
      - ./weights:/app/weights
    devices:
      - /dev/video0:/dev/video0  # Camera
      - /dev/ttyUSB0:/dev/ttyUSB0  # Robot
    privileged: true  # Required for hardware access
    restart: unless-stopped
    
  # Optional: Web interface or monitoring
  monitoring:
    image: grafana/grafana:latest
    container_name: tictactoe-monitoring
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-storage:/var/lib/grafana
    restart: unless-stopped

volumes:
  grafana-storage:
```

### 2. Container Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f tictactoe

# Update application
docker-compose pull
docker-compose up -d --force-recreate

# Shell access for debugging
docker-compose exec tictactoe bash
```

## Production Monitoring

### 1. Logging Configuration

#### Structured Logging
```python
import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        if hasattr(record, 'extra_data'):
            log_entry.update(record.extra_data)
            
        return json.dumps(log_entry)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/tictactoe/app.log'),
        logging.StreamHandler()
    ]
)
```

#### Log Rotation
```bash
# Configure logrotate
sudo tee /etc/logrotate.d/tictactoe > /dev/null << EOF
/var/log/tictactoe/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 644 tictactoe tictactoe
    postrotate
        systemctl reload tictactoe
    endscript
}
EOF
```

### 2. Health Monitoring

#### Health Check Endpoint
```python
from flask import Flask, jsonify
import threading
import time

app = Flask(__name__)

class HealthMonitor:
    def __init__(self):
        self.last_camera_frame = None
        self.last_robot_command = None
        self.detection_fps = 0
        
    def update_camera_status(self, frame_time):
        self.last_camera_frame = frame_time
        
    def update_robot_status(self, command_time):
        self.last_robot_command = command_time
        
    def get_health_status(self):
        current_time = time.time()
        
        camera_healthy = (
            self.last_camera_frame and 
            current_time - self.last_camera_frame < 5.0
        )
        
        robot_healthy = (
            self.last_robot_command and 
            current_time - self.last_robot_command < 30.0
        )
        
        return {
            'status': 'healthy' if camera_healthy and robot_healthy else 'unhealthy',
            'camera': {
                'status': 'healthy' if camera_healthy else 'unhealthy',
                'last_frame': self.last_camera_frame,
                'fps': self.detection_fps
            },
            'robot': {
                'status': 'healthy' if robot_healthy else 'unhealthy',
                'last_command': self.last_robot_command
            },
            'timestamp': current_time
        }

monitor = HealthMonitor()

@app.route('/health')
def health_check():
    return jsonify(monitor.get_health_status())

@app.route('/metrics')
def metrics():
    return jsonify({
        'detection_fps': monitor.detection_fps,
        'uptime': time.time() - app.start_time,
        'version': '1.0.0'
    })

if __name__ == '__main__':
    app.start_time = time.time()
    app.run(host='0.0.0.0', port=8080)
```

### 3. Performance Monitoring

#### Metrics Collection
```python
import psutil
import time
from dataclasses import dataclass
from typing import Dict

@dataclass
class PerformanceMetrics:
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    detection_fps: float
    processing_time_ms: float
    camera_fps: float
    robot_response_time_ms: float

class MetricsCollector:
    def __init__(self):
        self.process = psutil.Process()
        self.detection_times = []
        self.robot_times = []
        
    def collect_system_metrics(self) -> PerformanceMetrics:
        cpu_percent = self.process.cpu_percent()
        memory_info = self.process.memory_info()
        memory_percent = self.process.memory_percent()
        
        avg_detection_time = (
            sum(self.detection_times[-10:]) / len(self.detection_times[-10:])
            if self.detection_times else 0
        )
        
        avg_robot_time = (
            sum(self.robot_times[-10:]) / len(self.robot_times[-10:])
            if self.robot_times else 0
        )
        
        return PerformanceMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_mb=memory_info.rss / 1024 / 1024,
            detection_fps=1000 / avg_detection_time if avg_detection_time > 0 else 0,
            processing_time_ms=avg_detection_time,
            camera_fps=30,  # From camera controller
            robot_response_time_ms=avg_robot_time
        )
```

## Security Considerations

### 1. Network Security

#### Firewall Configuration
```bash
# Configure UFW (Ubuntu)
sudo ufw enable
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Allow SSH (if needed)
sudo ufw allow ssh

# Allow web interface (if enabled)
sudo ufw allow 8080/tcp

# Allow specific IPs only
sudo ufw allow from 192.168.1.0/24 to any port 8080
```

#### SSL/TLS Configuration
```python
# For web interfaces
import ssl
from flask import Flask

app = Flask(__name__)

# SSL context
context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
context.load_cert_chain('path/to/cert.pem', 'path/to/key.pem')

app.run(host='0.0.0.0', port=443, ssl_context=context)
```

### 2. Hardware Security

#### USB Device Restrictions
```bash
# Restrict USB devices to known hardware
sudo tee /etc/udev/rules.d/50-usb-security.rules > /dev/null << EOF
# Allow only specific robot
SUBSYSTEM=="usb", ATTR{idVendor}=="2341", ATTR{idProduct}=="0042", MODE="0664", GROUP="tictactoe"

# Allow only specific camera
SUBSYSTEM=="usb", ATTR{idVendor}=="046d", ATTR{idProduct}=="085b", MODE="0664", GROUP="video"

# Deny all other USB devices
SUBSYSTEM=="usb", MODE="0000"
EOF
```

## Troubleshooting

### 1. Common Issues

#### Camera Not Detected
```bash
# Check camera permissions
ls -la /dev/video*
groups $USER  # Should include 'video' group

# Test camera access
python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"

# Check USB bandwidth
lsusb -t
dmesg | grep -i usb
```

#### Robot Connection Issues
```bash
# Check serial permissions
ls -la /dev/ttyUSB*
groups $USER  # Should include 'dialout' group

# Test robot connection
python -c "
import pyuarm
arm = pyuarm.SwiftAPI()
print('Connecting...')
arm.connect()
print(f'Connected: {arm.connected}')
if arm.connected:
    pos = arm.get_position()
    print(f'Position: {pos}')
arm.disconnect()
"

# Check USB serial devices
dmesg | grep -i tty
```

#### Model Loading Errors
```bash
# Verify model files
ls -la weights/
file weights/*.pt

# Test model loading
python -c "
import torch
from ultralytics import YOLO
try:
    model = YOLO('weights/best_detection.pt')
    print('Detection model loaded successfully')
    model = YOLO('weights/best_pose.pt')
    print('Pose model loaded successfully')
except Exception as e:
    print(f'Error: {e}')
"
```

### 2. Performance Issues

#### GPU Acceleration
```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Install CUDA-enabled PyTorch (if needed)
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### Memory Optimization
```python
# Memory monitoring script
import psutil
import time

def monitor_memory():
    process = psutil.Process()
    while True:
        mem_info = process.memory_info()
        print(f"Memory: {mem_info.rss / 1024 / 1024:.1f} MB")
        time.sleep(5)

if __name__ == "__main__":
    monitor_memory()
```

### 3. Debugging Tools

#### Debug Scripts
```bash
# Test complete pipeline
python debug_coordinates.py

# Test grid detection only
python debug_grid_mapping.py

# Test camera resolution
python debug_resolution_mismatch.py

# Test robot movements
python -c "
from app.main.arm_controller import ArmController
arm = ArmController()
if arm.connect():
    arm.go_to_position(150, 0, 100)
    print('Movement test complete')
    arm.disconnect()
"
```

## Backup and Recovery

### 1. Configuration Backup

```bash
# Create backup script
cat > backup_config.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/backup/tictactoe/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Backup configuration
cp -r app/config/ "$BACKUP_DIR/"
cp app/calibration/hand_eye_calibration.json "$BACKUP_DIR/"

# Backup logs
cp -r /var/log/tictactoe/ "$BACKUP_DIR/logs/"

# Create archive
tar -czf "$BACKUP_DIR.tar.gz" -C "$BACKUP_DIR" .
rm -rf "$BACKUP_DIR"

echo "Backup created: $BACKUP_DIR.tar.gz"
EOF

chmod +x backup_config.sh
```

### 2. Recovery Procedures

```bash
# Restore from backup
BACKUP_FILE="backup_20231201_143000.tar.gz"
RESTORE_DIR="/tmp/restore"

mkdir -p "$RESTORE_DIR"
tar -xzf "$BACKUP_FILE" -C "$RESTORE_DIR"

# Restore configuration
cp -r "$RESTORE_DIR/config/" app/
cp "$RESTORE_DIR/hand_eye_calibration.json" app/calibration/

# Restart application
sudo systemctl restart tictactoe
```

---

This deployment guide provides comprehensive instructions for production deployment of the Robotic TicTacToe application with proper monitoring, security, and maintenance procedures.