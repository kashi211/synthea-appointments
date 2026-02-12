# Base image with Java (required by Synthea)
FROM eclipse-temurin:17-jdk

# Install Python
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

RUN chmod +x run_synthea generate_with_appointments.sh

# ENTRYPOINT allows argument passthrough
ENTRYPOINT ["bash", "generate_with_appointments.sh"]

# Defaults (can be overridden)
CMD ["10", "2"]
