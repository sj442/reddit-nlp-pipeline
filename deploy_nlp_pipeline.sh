#!/bin/bash
# ============================================================
#  Deploy Reddit NLP Pipeline to AWS ECR + ECS
# ============================================================

# ---- CONFIGURATION ----
AWS_REGION="us-east-1"               
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REPO_NAME="reddit-nlp-pipeline"       # ECR repo name
IMAGE_TAG="latest"                    
CLUSTER_NAME="reddit-nlp-cluster"     # ECS cluster name
SERVICE_NAME="reddit-nlp-service"     # ECS service name

# ---- STEP 1: Authenticate Docker to ECR ----
echo "🔑 Logging in to Amazon ECR..."
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# ---- STEP 2: Build Docker image ----
echo "🐳 Building Docker image..."
docker build -t $REPO_NAME .

# ---- STEP 3: Tag image ----
echo "🏷️ Tagging image..."
docker tag $REPO_NAME:$IMAGE_TAG $ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$REPO_NAME:$IMAGE_TAG

# ---- STEP 4: Push image to ECR ----
echo "🚀 Pushing image to ECR..."
docker push $ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$REPO_NAME:$IMAGE_TAG

# ---- STEP 5: Update ECS service to use the new image ----
echo "⚙️  Updating ECS service..."
aws ecs update-service \
  --cluster $CLUSTER_NAME \
  --service $SERVICE_NAME \
  --force-new-deployment \
  --region $AWS_REGION

echo "✅ Deployment complete! ECS is now pulling the latest image."
