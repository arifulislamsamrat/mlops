

# Dockerizing & Deploying ML Microservices in AWS

## Lab Overview

This comprehensive hands-on lab guides you through the complete lifecycle of building, containerizing, and deploying machine learning microservices. You'll develop two interconnected services, containerize them using Docker, and deploy them to AWS cloud infrastructure e.

### What You'll Learn

By completing this lab, you will:

- **Design and implement** microservices architecture with proper separation of concerns
- **Master Docker containerization** including multi-stage builds, networking, and orchestration
- **Deploy containerized applications** to AWS cloud infrastructure manually through the console
- **Implement comprehensive testing** strategies using Postman for API validation
- **Apply monitoring and troubleshooting** techniques for production-ready services
- **Understand DevOps workflows** from development to production deployment

## Prerequisites

- Python 3.8+ installed
- Docker and Docker Compose installed
- AWS account with CLI configured
- Postman installed
- Basic understanding of REST APIs and containers

---

## Step 1: Create Project Structure and Services

In this step, you'll establish the foundation of your microservices project by creating a well-organized directory structure. This organization is crucial for maintaining clean separation between services and supporting files. You'll also set up the basic framework that will house both of your microservices along with their deployment configurations.

### 1.1 Project Setup

```bash
mkdir ml_microservices_complete && cd ml_microservices_complete

# Create directory structure
mkdir -p service_a service_b postman_tests aws_deployment

```

## Architecture overview

![ml-microservices-svg (2).svg](ml-microservices-svg_(2).svg)

**Folder structure:**

```
ml_microservices_complete/
├── aws/
├── pulumi/
│   ├── __pycache__/
│   ├── venv/
│   ├── __main__.py              
│   ├── .gitignore
│   ├── poridhi-key               
│   ├── poridhi-key.pub          
│   ├── Pulumi.dev.yaml          
│   ├── Pulumi.yaml              
│   ├── README.md                
│   └── requirements.txt         
├── service_a/
│   ├── Dockerfile
│   ├── main.py
│   └── requirements.txt
├── service_b/
│   ├── Dockerfile
│   ├── main.py
│   └── requirements.txt
├── .dockerignore
├── awscliv2.zip                 
└── docker-compose.yml

```

---

## Step 2: Build Service A (Input Logger & API Gateway)

Service A will act as your main entry point and API gateway. In this step, you'll develop a sophisticated logging service that can intelligently route requests to your ML service when needed. The service will handle all incoming requests, log them for audit purposes, and decide whether to forward them to the machine learning service based on client preferences. This demonstrates the API gateway pattern commonly used in microservices architectures.

### 2.1 Create Service A Requirements

Create `service_a/requirements.txt`:

```
fastapi==0.103.1
uvicorn==0.23.2
requests==2.31.0
pydantic==2.4.2
```

### 2.2 Implement Service A

Create `service_a/main.py`:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import logging
import uvicorn
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("service_a")

app = FastAPI(
    title="Service A - Input Logger & API Gateway",
    description="Logs input data and forwards ML prediction requests",
    version="1.0.0"
)

# Define input data model
class InputData(BaseModel):
    data: str
    forward_to_model: bool = True

class HealthResponse(BaseModel):
    service: str
    status: str
    timestamp: str
    version: str

# Service B URL - configurable for different environments
SERVICE_B_URL = os.getenv("SERVICE_B_URL", "http://localhost:8001/predict")

@app.get("/", response_model=dict)
def read_root():
    return {
        "service": "Service A - Input Logger & API Gateway",
        "status": "running",
        "endpoints": ["/health", "/process"],
        "version": "1.0.0"
    }

@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        service="Service A",
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )

@app.post("/process")
async def process_input(input_data: InputData):
    # Log the received input with timestamp
    timestamp = datetime.now().isoformat()
    logger.info(f"[{timestamp}] Received input: {input_data.data}")

    # If forward_to_model is True, send the data to Service B
    if input_data.forward_to_model:
        try:
            logger.info(f"Forwarding request to Service B at: {SERVICE_B_URL}")
            response = requests.post(
                SERVICE_B_URL,
                json={"input": input_data.data},
                timeout=10
            )
            response.raise_for_status()

            # Return both the logged status and the prediction from Service B
            ml_response = response.json()
            return {
                "status": "Input logged successfully",
                "timestamp": timestamp,
                "service": "Service A",
                "model_prediction": ml_response,
                "forwarded_to": SERVICE_B_URL
            }

        except requests.RequestException as e:
            error_msg = f"Service B is unavailable: {str(e)}"
            logger.error(error_msg)
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "Service B unavailable",
                    "message": error_msg,
                    "timestamp": timestamp,
                    "service_b_url": SERVICE_B_URL
                }
            )
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Internal server error",
                    "message": str(e),
                    "timestamp": timestamp
                }
            )
    else:
        # Just return the logged status
        return {
            "status": "Input logged successfully",
            "timestamp": timestamp,
            "service": "Service A",
            "forwarded": False
        }

@app.get("/metrics")
def get_metrics():
    """Simple metrics endpoint for monitoring"""
    return {
        "service": "Service A",
        "uptime": "available",
        "service_b_configured": SERVICE_B_URL,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )

```

### Let discuss shortly what each of the line does-

**Imports**: Brings in FastAPI framework, data validation tools, HTTP client, logging utilities, the ASGI server, and datetime functionality.

**Logging Setup**: Configures application-wide logging with timestamps and creates a dedicated logger for this service.

**App Initialization**: Creates the FastAPI application with metadata for auto-generated documentation.

**Data Models**: Defines two validation models - one for incoming requests (with data and a forwarding flag) and another for health check responses.

**Service Configuration**: Sets up the URL for Service B (the ML prediction service) from environment variables.

**Endpoints**:

- **Root endpoint**: Provides service information and lists available endpoints
- **Health check**: Returns current service status with timestamp
- **Process endpoint**: The main functionality - logs incoming data and optionally forwards it to Service B for ML predictions. Includes error handling for service unavailability
- **Metrics endpoint**: Exposes basic monitoring information

**Error Handling**: Manages two types of failures - Service B unavailability (returns 503) and unexpected errors (returns 500).

**Server Configuration**: Starts the Uvicorn ASGI server on a configurable port, listening on all network interfaces.

The service essentially acts as a middleware layer that logs all requests and can route them to a machine learning service while providing proper error handling and monitoring capabilities.

### 2.3 Create Dockerfile for Service A

Create `service_a/Dockerfile`:

```
# Use Python slim image for smaller size
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies including curl for health checks
RUN apt-get update && \
    apt-get install -y curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "main.py"]

```

---

### Here's what each section does:-

**Base Image**: Uses Python 3.9 slim version for a smaller container size while still having Python runtime.

**Working Directory**: Sets `/app` as the working directory where all subsequent commands will be executed.

**System Dependencies**: Updates package lists and installs curl (needed for health checks), then cleans up package caches to reduce image size.

**Requirements Installation**: Copies only the requirements.txt file first and installs Python dependencies. This leverages Docker's layer caching - if requirements don't change, this layer can be reused.

**Application Code**: Copies all application files into the container.

**Security Setup**: Creates a non-root user called 'appuser' with UID 1000, changes ownership of the app directory to this user, and switches to run as this user for better security.

**Port Configuration**: Declares that the container will listen on port 8000 (documentation purposes).

**Health Check**: Configures Docker to periodically check if the service is healthy by calling the `/health` endpoint. It checks every 30 seconds, waits 10 seconds for response, allows 5 seconds for startup, and retries 3 times before marking as unhealthy.

**Startup Command**: Runs the Python application using the main.py file when the container starts.

This configuration creates a secure, optimized container that runs the FastAPI service with built-in health monitoring.

## Step 3: Build Service B (ML Prediction Service)

In this step, you'll create the machine learning prediction service that serves as the brain of your microservices architecture. Service B will implement an advanced ML model simulation that processes text inputs and returns classification predictions with confidence scores. The service demonstrates how to build a dedicated microservice focused on a single responsibility while providing comprehensive error handling and monitoring capabilities.

### 3.1 Create Service B Requirements

Create `service_b/requirements.txt`:

```
fastapi==0.103.1
uvicorn==0.23.2
pydantic==2.4.2
scikit-learn==1.3.2
numpy==1.24.3

```

### 3.2 Implement Service B

Create `service_b/main.py`:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import uvicorn
import random
import numpy as np
import os
from datetime import datetime
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("service_b")

app = FastAPI(
    title="Service B - ML Prediction Service",
    description="Machine Learning prediction service with classification capabilities",
    version="1.0.0"
)

# Define input/output data models
class PredictionRequest(BaseModel):
    input: str

class PredictionResult(BaseModel):
    class_: str
    confidence: float
    input_length: int
    processing_time_ms: float

class PredictionResponse(BaseModel):
    prediction: PredictionResult
    message: str
    timestamp: str
    model_version: str

class HealthResponse(BaseModel):
    service: str
    status: str
    model_status: str
    timestamp: str
    version: str

# Enhanced ML Model simulation
class AdvancedMLModel:
    def __init__(self):
        self.classes = ["cat", "dog", "bird", "fish", "rabbit", "horse", "elephant", "tiger"]
        self.model_version = "v2.1.0"
        # Fixed: Create feature weights that match the number of features (not classes)
        # We extract 5 features, so we need 5 weights
        self.feature_weights = np.random.random(5)  # Changed from len(self.classes) to 5
        logger.info(f"Advanced ML model initialized - Version: {self.model_version}")
        logger.info(f"Available classes: {self.classes}")

    def extract_features(self, input_text: str) -> np.ndarray:
        """Extract features from input text (simulated)"""
        # Simple feature extraction based on text characteristics
        features = [
            len(input_text),  # Length
            input_text.count(' '),  # Word count
            len(set(input_text.lower())),  # Unique characters
            input_text.count('a') + input_text.count('e'),  # Vowel count
            hash(input_text.lower()) % 100,  # Text hash feature
        ]

        # Normalize features to prevent overflow
        features = np.array(features, dtype=float)
        # Add small epsilon to prevent division by zero
        norm = np.linalg.norm(features) + 1e-8
        features = features / norm
        return features

    def predict(self, input_text: str) -> Dict[str, Any]:
        """Make prediction with enhanced logic"""
        start_time = datetime.now()

        logger.info(f"Processing prediction for input: '{input_text[:50]}...'")

        # Extract features
        features = self.extract_features(input_text)

        # Simulate model computation
        text_lower = input_text.lower()

        # Enhanced prediction logic based on keywords
        class_probabilities = {}

        for i, class_name in enumerate(self.classes):
            # Base probability
            prob = 0.1 + random.random() * 0.3

            # Boost probability if class name appears in text
            if class_name in text_lower:
                prob += 0.4

            # Feature-based adjustment - Fixed the dimension mismatch
            # Use sum of features as a single score instead of dot product
            feature_score = np.sum(features) * 0.1
            prob += feature_score

            class_probabilities[class_name] = prob

        # Normalize probabilities
        total_prob = sum(class_probabilities.values())
        class_probabilities = {k: v/total_prob for k, v in class_probabilities.items()}

        # Select predicted class
        predicted_class = max(class_probabilities, key=class_probabilities.get)
        confidence = class_probabilities[predicted_class]

        # Ensure confidence is realistic (0.6-0.99)
        confidence = max(0.6, min(0.99, confidence))

        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        logger.info(f"Prediction completed: {predicted_class} (confidence: {confidence:.2f})")

        return {
            "class": predicted_class,
            "confidence": round(confidence, 3),
            "input_length": len(input_text),
            "processing_time_ms": round(processing_time, 2),
            "model_version": self.model_version
        }

# Initialize the ML model
model = AdvancedMLModel()

@app.get("/", response_model=dict)
def read_root():
    return {
        "service": "Service B - ML Prediction Service",
        "status": "running",
        "model_version": model.model_version,
        "available_classes": model.classes,
        "endpoints": ["/health", "/predict", "/model_info"],
        "version": "1.0.0"
    }

@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        service="Service B",
        status="healthy",
        model_status="ready",
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make ML prediction on input data"""
    try:
        # Validate input
        if not request.input or len(request.input.strip()) == 0:
            raise HTTPException(
                status_code=400,
                detail="Input cannot be empty"
            )

        # Get prediction from model
        prediction_result = model.predict(request.input)

        # Create response
        prediction = PredictionResult(
            class_=prediction_result["class"],
            confidence=prediction_result["confidence"],
            input_length=prediction_result["input_length"],
            processing_time_ms=prediction_result["processing_time_ms"]
        )

        response = PredictionResponse(
            prediction=prediction,
            message=f"Predicted class: {prediction_result['class']} with {prediction_result['confidence']*100:.1f}% confidence",
            timestamp=datetime.now().isoformat(),
            model_version=prediction_result["model_version"]
        )

        return response

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Prediction failed",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

@app.get("/model_info")
def get_model_info():
    """Get information about the ML model"""
    return {
        "model_version": model.model_version,
        "available_classes": model.classes,
        "model_type": "Enhanced Classification Model",
        "features": "Text-based feature extraction",
        "last_updated": datetime.now().isoformat()
    }

@app.get("/metrics")
def get_metrics():
    """Simple metrics endpoint for monitoring"""
    return {
        "service": "Service B",
        "model_version": model.model_version,
        "available_classes_count": len(model.classes),
        "uptime": "available",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8001))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )
```

### Here's what each section does:

**Imports and Setup**: Brings in FastAPI, data validation, logging, numerical computing with NumPy, and other utilities needed for the ML service.

**Logging Configuration**: Sets up structured logging with timestamps for tracking prediction requests and model operations.

**App Initialization**: Creates the FastAPI instance with metadata describing it as an ML prediction service.

**Data Models**: Defines Pydantic models for:

- **PredictionRequest**: Accepts input text for classification
- **PredictionResult**: Contains the predicted class, confidence score, input length, and processing time
- **PredictionResponse**: Wraps the prediction result with metadata
- **HealthResponse**: Structures health check information

**AdvancedMLModel Class**: Simulates an ML model with:

- **Initialization**: Sets up 8 animal classes, creates random feature weights, and logs model info
- **Feature Extraction**: Extracts 5 features from text (length, word count, unique characters, vowel count, hash value) and normalizes them
- **Prediction Logic**:
    - Calculates base probabilities for each class
    - Boosts probability if class name appears in input text
    - Adjusts based on extracted features
    - Normalizes probabilities and selects highest confidence class
    - Ensures realistic confidence scores (60-99%)

**Endpoints**:

- **Root endpoint**: Returns service info, model version, and available classes
- **Health check**: Confirms service and model are ready
- **Predict endpoint**: Main functionality - validates input, gets prediction from model, formats response with confidence percentage
- **Model info**: Provides details about the ML model
- **Metrics**: Exposes monitoring information

**Error Handling**: Catches empty inputs (400 error) and prediction failures (500 error) with detailed error messages.

**Server Configuration**: Runs on port 8001 by default, configurable via environment variable.

The service simulates a text classification model that predicts animal classes based on input text, with enhanced logic that considers keyword matching and text features to generate confidence scores.

### 3.3 Create Dockerfile for Service B

Create `service_b/Dockerfile`:

```
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y curl gcc && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

CMD ["python", "main.py"]

```

---

This Dockerfile configures the ML prediction service container. Here's what each section does:

**Base Image**: Uses Python 3.9 slim for a lightweight container.

**Working Directory**: Sets `/app` as the working directory.

**System Dependencies**: Installs curl (for health checks) and gcc (required for compiling some Python packages like NumPy). Cleans up afterward to minimize image size.

**Python Dependencies**: Copies requirements.txt first and installs packages separately for better Docker layer caching.

**Application Code**: Copies all application files into the container.

**Security Configuration**: Creates a non-root user 'appuser' with UID 1000, changes ownership of the app directory, and switches to this user for improved security.

**Port Declaration**: Exposes port 8001 (Service B's default port).

**Health Check**: Configures Docker to monitor service health by calling the `/health` endpoint every 30 seconds, with a 10-second timeout, 5-second startup period, and 3 retry attempts.

**Startup Command**: Runs the Python application when the container starts.

The key difference from Service A's Dockerfile is the addition of gcc in system dependencies (needed for NumPy compilation) and the use of port 8001 instead of 8000. This creates a secure, optimized container for the ML prediction service with automated health monitoring.

## Step 4: Docker Compose Configuration

Now you'll create the orchestration layer that brings both services together into a cohesive system. Docker Compose will manage the networking between your services, handle startup dependencies, and provide configuration management. You'll create two separate compose files to handle different deployment scenarios - one optimized for local development with debugging capabilities, and another configured for production deployment with appropriate resource constraints.

### 4.1 Create .dockerignore

Create `.dockerignore`:

```
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.venv/
.git/
.gitignore
*.md
.pytest_cache/
.coverage
node_modules/
.DS_Store
.env
docker-compose*.yml
Dockerfile*
.dockerignore
```

### 4.2 Local Development Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  service-a:
    build:
      context: ./service_a
      dockerfile: Dockerfile
    container_name: ml-service-a
    ports:
      - "8000:8000"
    environment:
      - SERVICE_B_URL=http://service-b:8001/predict
      - LOG_LEVEL=INFO
      - PORT=8000
    depends_on:
      service-b:
        condition: service_healthy
    networks:
      - ml-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  service-b:
    build:
      context: ./service_b
      dockerfile: Dockerfile
    container_name: ml-service-b
    ports:
      - "8001:8001"
    environment:
      - LOG_LEVEL=INFO
      - PORT=8001
    networks:
      - ml-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8001/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

networks:
  ml-network:
    driver: bridge
    name: ml-microservices-network

volumes:
  app-data:
    name: ml-microservices-data
```

This Docker Compose file orchestrates both microservices. Here's what each section does:

**Version**: Specifies Docker Compose file format version 3.8.

**Services Section**:

**Service A Configuration**:

- **Build**: Builds from the Dockerfile in `./service_a` directory
- **Container Name**: Names the container `ml-service-a`
- **Ports**: Maps host port 8000 to container port 8000
- **Environment Variables**:
    - Sets Service B's URL for internal communication
    - Configures logging level and port
- **Dependencies**: Waits for Service B to be healthy before starting
- **Networks**: Connects to the shared `ml-network`
- **Restart Policy**: Restarts unless manually stopped
- **Health Check**: Monitors service health with 30-second intervals

**Service B Configuration**:

- **Build**: Builds from the Dockerfile in `./service_b` directory
- **Container Name**: Names the container `ml-service-b`
- **Ports**: Maps host port 8001 to container port 8001
- **Environment Variables**: Sets logging level and port
- **Networks**: Connects to the same `ml-network`
- **Restart Policy**: Restarts unless manually stopped
- **Health Check**: Similar health monitoring configuration

**Networks Section**:

- Creates a bridge network named `ml-microservices-network` for inter-service communication
- Allows services to communicate using container names as hostnames

**Volumes Section**:

- Defines a named volume `ml-microservices-data` (though not currently mounted by either service)

This configuration ensures Service A can communicate with Service B using the internal network, manages service startup order with health checks, and provides automatic restart capabilities for resilience.

## Step 4: Local Testing

Before moving to cloud deployment, we'll validate your containerized microservices in your local environment. This step involves building Docker images, starting the services with Docker Compose, and performing comprehensive testing to ensure proper communication between services. You'll verify health endpoints, test the logging functionality, and confirm the complete ML pipeline works as expected.

### 5.1 Build and Run Locally

```bash
# Build images
docker-compose build

# Start services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f

# Test services
curl http://localhost:8000/health
curl http://localhost:8001/health

```

After docker-compose build ,we will start our services .And we will get a terminal prompt to goto port 8000 and 8001. 

![docker up](image.png)

By using docker-compose ps we will see detail information -

![docker ps](image%201.png)

### 5.2 Test the Pipeline

```bash
# Test Service A without ML forwarding
curl -X POST -H "Content-Type: application/json" \
  -d '{"data": "Local test data", "forward_to_model": false}' \
  http://localhost:8000/process

# Test Service B directly
curl -X POST -H "Content-Type: application/json" \
  -d '{"input": "Direct ML test input"}' \
  http://localhost:8001/predict
  
  # Test full ML pipeline
curl -X POST -H "Content-Type: application/json" \
  -d '{"data": "Local test data", "forward_to_model": false}' \
  http://localhost:8000/process

```

---

When we test the service A without ML forwarding it will be resulted like this -

![local-curl-8000](image%202.png)

for service B-

![local-curl-service-b](image%203.png)

Result of ML pipeline -

![pipeline-curl](image%204.png)

## Step 06: AWS deployment with pulumi

---

**What is Pulumi?**
Pulumi is an Infrastructure as Code (IaC) tool that lets you define cloud infrastructure using familiar programming languages like Python, TypeScript, Go, and C# instead of YAML or JSON templates.

**Key Benefits for ML Microservices**

**Language Flexibility**: Write infrastructure code in the same language as your ML models, enabling better integration between application and infrastructure code.

**AWS Integration**: Native support for AWS services commonly used in ML deployments including ECS/EKS for container orchestration, Lambda for serverless inference, SageMaker for managed ML services, and API Gateway for model endpoints.

**Microservices Architecture Support**: Easily define multiple related services with shared resources like VPCs, security groups, and load balancers while maintaining service isolation.

**State Management**: Automatic tracking of infrastructure state with support for team collaboration through Pulumi Cloud or self-hosted backends.

**Configuration Management**: Built-in secrets management and environment-specific configurations for dev/staging/production deployments.

**Example Use Cases**

- Deploy containerized ML models to ECS Fargate with auto-scaling
- Set up API Gateway + Lambda for lightweight model inference
- Create SageMaker endpoints with custom Docker images
- Orchestrate multiple model services with shared networking and monitoring

### Step 6.1: Install Prerequisites

**Install Python and pip**:

Ensure Python 3.9+ and pip are installed:

```bash
sudo apt update
sudo apt install python3.8-venv
sudo apt install python3-pip -y
python3 --version
pip3 --version
```

If not installed, install them using your package manager.

**Install Pulumi CLI**:

```bash
curl -fsSL https://get.pulumi.com | sh
```

After installation, restart your terminal or run:

```bash
export PATH=$PATH:$HOME/.pulumi/bin
```

**Install AWS CLI**:

```bash
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
sudo apt install unzip -y
unzip awscliv2.zip
sudo ./aws/install
```

## Step 6.2 : Configure AWS Credentials

Using the credentials provided by Poridhi:

```bash
aws configure
```

Enter the following when prompted:

- **AWS Access Key ID**: `<Your AccessKey>`
- **AWS Secret Access Key**: `<Your SecretKey>`
- **Default region name**: `ap-southeast-1`
- **Default output format**: `json`

This will create the necessary configuration files in `~/.aws/`.

---

### Step 6.3: Initialize Pulumi Project

**Navigate to your project directory**:

```bash
cd ml_microservices_complete
```

**Create a new directory for Pulumi configurations**:

```bash
mkdir pulumi
cd pulumi
```

1. **Initialize a new Pulumi project**:
    
    ```bash
    pulumi new aws-python
    ```
    
    If You don’t have pulumi account open it And create a access token 
    
    ![pulumi-setup-aws.png](image%205.png)
    
    click that link and give a description and create access token
    
    ![pulumi-token.png](image%206.png)
    
    After entering the token and  When prompted:
    
    - **Project name**: `ml-microservices`
    - **Project description**: `Dockerizing & Deploying ML Microservices in AWS with Pulumi`
    - **Stack name**: `dev`
    - **AWS region**: `ap-southeast-1`

     toolchain should  select pip and then use your region

![toolchain-pip-select.png](image%207.png)

1. **Install Python dependencies**:
    
    Create a `requirements.txt` file in pulumi folder with the following content:
    
    ```
    pulumi
    pulumi_aws
    pulumi_docker
    ```
    
    Then install the dependencies:
    
    ```bash
    pip3 install -r requirements.txt
    ```
    

## Step 6.4: Define Infrastructure with Pulumi

In the `pulumi/__main__.py` file, define the necessary AWS resources:

```python
import pulumi
import pulumi_aws as aws
import base64

# VPC
vpc = aws.ec2.Vpc("my-vpc", cidr_block="10.0.0.0/16")

# Subnet
subnet = aws.ec2.Subnet("my-subnet",
    cidr_block="10.0.2.0/24",
    vpc_id=vpc.id,
    availability_zone="ap-southeast-1a")

# Internet Gateway
igw = aws.ec2.InternetGateway("my-igw", vpc_id=vpc.id)

# Route Table & Association
route_table = aws.ec2.RouteTable("my-route-table", vpc_id=vpc.id,
    routes=[{"cidr_block": "0.0.0.0/0", "gateway_id": igw.id}])

route_table_assoc = aws.ec2.RouteTableAssociation("my-route-table-assoc",
    subnet_id=subnet.id,
    route_table_id=route_table.id)

# Security Group (Open ports 22, 80, 8000, 8001)
security_group = aws.ec2.SecurityGroup("web-sg",
    description="Allow SSH and Web",
    vpc_id=vpc.id,
    ingress=[
        {"protocol": "tcp", "from_port": 22, "to_port": 22, "cidr_blocks": ["0.0.0.0/0"]},
        {"protocol": "tcp", "from_port": 8000, "to_port": 8000, "cidr_blocks": ["0.0.0.0/0"]},
        {"protocol": "tcp", "from_port": 8001, "to_port": 8001, "cidr_blocks": ["0.0.0.0/0"]},
    ],
    egress=[{"protocol": "-1", "from_port": 0, "to_port": 0, "cidr_blocks": ["0.0.0.0/0"]}]
)

# SSH Key
with open("poridhi-key.pub") as f:
    public_key = f.read()

key_pair = aws.ec2.KeyPair("poridhi-key", public_key=public_key)

# EC2 User Data
user_data_script = """#!/bin/bash
apt update
apt install -y docker.io git
systemctl start docker
systemctl enable docker
usermod -aG docker ubuntu
"""

instance = aws.ec2.Instance("ml-instance",
    ami="ami-0fa377108253bf620",  # Ubuntu 22.04 (Singapore region)
    instance_type="t2.micro",
    vpc_security_group_ids=[security_group.id],
    subnet_id=subnet.id,
    associate_public_ip_address=True,
    key_name=key_pair.key_name,
    user_data=user_data_script,
    tags={"Name": "ML-Microservice-Instance"}
)

pulumi.export("public_ip", instance.public_ip)
pulumi.export("public_dns", instance.public_dns)

```

---

This Pulumi script automatically creates a complete AWS setup to run your machine learning services in the cloud. It builds a secure network, sets up firewall rules to allow access to your ML APIs on ports 8000-8001, and launches an Ubuntu server that automatically installs Docker and downloads your ML code from GitHub. When the server starts up, it runs a script that pulls your microservices repository and starts all your ML services using Docker Compose. The whole process takes about 5-10 minutes, and you get a public IP address where people can access your ML models through web APIs. This eliminates all the manual AWS console clicking and gives you a reproducible way to deploy ML services. It's perfect for turning your local ML experiments into live web services that anyone can use.

**Now create a public key to access** 

```python
ssh-keygen -t rsa -b 2048 -f poridhi-key # 
```

### What This Command Does

- Creates two files: `poridhi-key` (private key) and `poridhi-key.pub` (public key)
- The Pulumi script reads the `.pub` file to set up AWS access
- You use the private key file to SSH into your server later

## Step 6.5 : Deploy Infrastructure

**Preview the deployment**:

```bash
pulumi preview
```

So after previewing we wiill see these preview 

![pulumi-preview.png](image%208.png)

**Deploy the infrastructure**:

```bash
pulumi up
```

Confirm the deployment when prompted. Then make sure you entered the update . It will give us the public ip 

![public-ip.png](image%209.png)

Connecting with ssh

```python
chmod 400 poridhi-key
ssh -i poridhi-key ubuntu@<your-public-ip>  #use public-key you got from pulumi up
```

So now we are on the AWS cli

![aws-console.png](image%2010.png)

Now we are all set we will install docker

```python
# 1. Update package index
sudo apt update

# 2. Install required packages
sudo apt install -y apt-transport-https ca-certificates curl software-properties-common

# 3. Add Docker’s official GPG key
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# 4. Add Docker repository
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] \
https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | \
sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# 5. Update the package index again
sudo apt update

# 6. Install Docker
sudo apt install -y docker-ce docker-ce-cli containerd.io

# 7. Verify Docker is installed
docker --version

```

now use these command 

```python
git clone https://github.com/arifulislamsamrat/ml_microservices_complete.git
sudo docker-compose build
sudo docker-compose up -d
```

This will clone our repository and dockerize all services .we can check with ***sudo docker-compose ps***

![aws-cli-docker.png](image%2011.png)

# Accessing Your Deployed Services Using Postman

1. **Open Postman.**
2. **Go to** the desired **workspace** (or create a new one).
3. For testing your deployed services:
    - To test **`service_a`**, use:
        
        ```
        Method: GET
        URL: http://public_ip_of_your_AWS_ubuntu_instance:8000/ 
        ```
        
    
    ![postman-service-a.png](image%2012.png)
    
    After it we will see the responses. And If I want to check health we will send /health 
    
    ![postman-service-a-health.png](image%2013.png)
    
    - To test **`service_b`**, use:
        
        ```
        makefile
        CopyEdit
        Method: GET
        URL: http://public_ip_of_your_AWS_ubuntu_instance:8001/ 
        
        ```
        
4. **Click "Send"** to see the response from each service.

![postmen-service-b.png](image%2014.png)

To see the  model_info    we will use http://public_ip_of_your_AWS_ubuntu_instance:8001/model_info 

![postmen-service-b-model-info.png](8bd27b95-54ab-4f3a-8379-b086269ffb36.png)

If forget your public ip then use ***ifconfig.me***  

### Container Monitoring

Monitor your container status and resource usage using Docker commands. Check the running status of all containers and review their resource consumption patterns. Use docker-compose ps to see the current state of your services and docker stats to monitor real-time resource usage including CPU, memory, and network metrics.

```bash
docker-compose ps
docker-compose logs -f service-a
docker-compose logs -f service-b
docker stats

```

Examine container health status by inspecting the health check results. Docker's built-in health monitoring provides valuable insights into service availability and can help identify issues before they impact users.

```bash
docker inspect ml-service-a | grep Health
docker inspect ml-service-b | grep Health

```

### 10.2 Network Troubleshooting

Diagnose network connectivity issues by testing inter-service communication directly. Use docker-compose exec to run commands inside containers and verify that services can reach each other using their container names.

```bash
docker-compose exec service-a curl http://service-b:8001/health
docker network ls
docker network inspect ml-microservices-network

```

Test external connectivity to ensure your services are accessible from outside the container network. This helps identify port mapping issues or firewall problems that might prevent client access.

```bash
curl -v http://localhost:8000/health

```

### 10.3 Application Monitoring

Monitor application-specific metrics by examining logs for processing times, error rates, and request patterns. Look for performance bottlenecks or unusual error patterns that might indicate issues with your ML processing or service communication.

```bash
docker-compose logs service-a | grep "processing"
docker-compose logs service-b | grep "prediction"

```

Measure response times for your endpoints to ensure they meet performance expectations. Use timing tools to monitor how long requests take to complete, which helps identify performance degradation over time.

```bash
time curl http://localhost:8000/process -X POST \
  -H "Content-Type: application/json" \
  -d '{"data": "performance test", "forward_to_model": true}'

```

---

## Conclusion

We've successfully built a complete containerized ML microservices solution from scratch and deployed it to AWS cloud platform. Throughout this comprehensive lab, you've gained hands-on experience with modern containerization technologies, microservices architecture patterns, and cloud deployment strategies.

Your journey began with developing two independent services that demonstrate proper separation of concerns and service communication patterns. You then containerized these services using Docker, learning how to create efficient, secure, and maintainable container images. The Docker Compose orchestration layer taught you how to manage multi-container applications with proper networking, dependencies, and configuration management.

The AWS deployment phase demonstrated how containerized applications provide deployment consistency across different environments. Your services now run reliably in the cloud with the same behavior and performance characteristics they exhibited in your local development environment. The comprehensive Postman testing suite ensures that your services maintain their functionality and reliability regardless of where they're deployed.

The monitoring and troubleshooting techniques you've learned provide the foundation for maintaining production services. You now understand how to diagnose issues, monitor performance, and ensure your microservices continue operating effectively under various conditions.

This containerized architecture serves as a solid foundation for building larger, more complex microservices ecosystems. The patterns and practices you've implemented here scale effectively and can be extended to support additional services, databases, message queues, and other components commonly found in production microservices architectures.