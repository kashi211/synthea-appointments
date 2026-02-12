# Base image with Java (required by Synthea)
FROM eclipse-temurin:17-jdk

# Install Python
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy entire project
COPY . .

# Make scripts executable
RUN chmod +x run_synthea generate_with_appointments.sh

# Default command
CMD ["bash", "generate_with_appointments.sh"]
