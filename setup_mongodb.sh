#!/bin/bash

echo "🚀 VietMind AI - MongoDB Setup"
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running"
    echo ""
    echo "Please choose an option:"
    echo "1. Start Docker Desktop and run this script again"
    echo "2. Use MongoDB Atlas (cloud) - Update MONGODB_URL in .env"
    echo "   Example: mongodb+srv://username:password@cluster.mongodb.net/vietmind_ai"
    echo ""
    exit 1
fi

# Check if MongoDB container exists
if docker ps -a | grep -q mongodb; then
    echo "📦 MongoDB container already exists"

    # Check if it's running
    if docker ps | grep -q mongodb; then
        echo "✅ MongoDB is already running"
    else
        echo "🔄 Starting existing MongoDB container..."
        docker start mongodb
        echo "✅ MongoDB started successfully"
    fi
else
    echo "📦 Creating new MongoDB container..."
    docker run -d \
        -p 27017:27017 \
        --name mongodb \
        -e MONGO_INITDB_DATABASE=vietmind_ai \
        mongo:latest

    echo "✅ MongoDB container created and running"
fi

echo ""
echo "🎉 MongoDB is ready!"
echo "   URL: mongodb://localhost:27017"
echo "   Database: vietmind_ai"
echo ""
echo "To stop MongoDB: docker stop mongodb"
echo "To view logs: docker logs mongodb"
