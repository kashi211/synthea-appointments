

# 🏥 Synthea + FHIR Appointment Generator (Dockerized)

This project generates **synthetic FHIR patient data** using **Synthea**, then enriches it with **Appointment resources**, keeping bundle sizes controlled.

Everything runs **inside Docker** — no local Java or Python setup required.

---

## ✅ Prerequisites (One-Time)

Your system must have:

* **macOS**
* **Docker Desktop**
  👉 [https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/)

Verify installation:

```bash
docker --version
docker compose version
```

---

## 📥 Step 1 — Clone the Repository

```bash
git clone <REPO_URL>
cd synthea-appointments
```

(Replace `<REPO_URL>` with the actual repository URL)

---

## 🏗 Step 2 — Build the Docker Image (One-Time)

This builds Synthea + Python + the appointment enrichment logic.

```bash
docker compose build
```

> ⚠️ You only need to do this **once**, unless the code changes.

---

## ▶️ Step 3 — Run Data Generation

Run Synthea and add Appointment resources using:

```bash
docker compose run synthea 50 20
```

### What this means:

| Argument | Meaning                                  |
| -------- | ---------------------------------------- |
| `50`     | Number of patients to generate           |
| `20`     | Maximum bundle size **per patient** (MB) |


---

## 📂 Output Location

FHIR bundles are written to:

```
output/fhir/
```

Each file represents **one patient** and includes:

* Patient
* Encounter
* Condition
* Observation
* MedicationRequest
* DiagnosticReport
* DocumentReference
* Provenance
* Appointment (controlled volume)

---

## 🔁 Re-Running the Generator

To generate fresh data again:

```bash
docker compose run synthea 10 2
```

You **do not need to rebuild** unless code changes.
