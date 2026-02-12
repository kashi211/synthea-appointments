
#!/usr/bin/env python3
"""
Add FHIR Appointment resources to Synthea-generated patient bundles.
Synthea does not emit Appointment resources; this script creates them from
existing Encounters. For each encounter we emit 2–3 Appointments (dynamic)
modeling the lifecycle: BOOKED → ARRIVED → FULFILLED, all linked via basedOn.

Appointment volume is controlled by:
  1) Encounter class filter: only AMB (ambulatory); emergency/inpatient/urgent skipped.
  2) Probability: only a fraction of qualifying encounters get appointments (default 40%).
  3) Per-patient cap: max appointments per patient (default 12).

Usage:
  python3 add_appointments.py -p 10     # Run Synthea for 10 patients, then add Appointments
  python3 add_appointments.py --no-synthea   # Only add Appointments to existing output/fhir/*.json
  python3 add_appointments.py --max-bundle-size 4   # Delete patient bundles larger than 4 MB (omit to keep all)
"""

import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Default paths relative to script location
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_FHIR = SCRIPT_DIR / "output" / "fhir"
RUN_SYNTHEA = SCRIPT_DIR / "run_synthea"

# Patient bundle files have pattern: Name_Name_uuid.json (exclude hospital/practitioner info)
BUNDLE_PATTERN = re.compile(r"^[^_]+_[^_]+_.*\.json$")
SKIP_PREFIXES = ("hospitalInformation", "practitionerInformation")

# Appointment statuses - weighted for realistic distribution
STATUS_WEIGHTS = [
    ("fulfilled", 72),
    ("cancelled", 10),
    ("noshow", 6),
    ("booked", 5),
    ("arrived", 4),
    ("pending", 3),
]

# Cancellation reasons when status is cancelled or noshow
CANCEL_REASONS = [
    {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/appointment-cancellation-reason", "code": "pat", "display": "Patient"}]},
    {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/appointment-cancellation-reason", "code": "pat-crs", "display": "Patient: Canceled via patient portal"}]},
    {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/appointment-cancellation-reason", "code": "prov", "display": "Provider"}]},
    {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/appointment-cancellation-reason", "code": "maint", "display": "Equipment Maintenance"}]},
    {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/appointment-cancellation-reason", "code": "other", "display": "Other"}]},
]

DURATION_MINUTES = [15, 20, 30, 45, 60]
APPT_OFFSET_MINUTES = (0, 15, 30)  # appointment start before encounter (min, max)

# Option 1: Only create appointments for these encounter classes (ambulatory/outpatient; skip emergency/inpatient/urgent)
ENCOUNTER_CLASS_CODES_FOR_APPOINTMENTS = {"AMB"}  # v3-ActCode: AMB=ambulatory
# Option 2: Probability that a qualifying encounter gets appointments (e.g. 0.4 = 40%)
APPOINTMENT_PROBABILITY = 0.4
# Option 3: Per-patient cap is randomized for variety (min–max range)
MIN_APPOINTMENTS_PER_PATIENT = 1
MAX_APPOINTMENTS_PER_PATIENT = 12

def _clear_fhir_output(fhir_dir: Path) -> None:
    """Remove all files and subdirectories in output/fhir (clean slate before Synthea run)."""
    if not fhir_dir.exists() or not fhir_dir.is_dir():
        return
    for p in fhir_dir.iterdir():
        if p.is_file():
            p.unlink()
        else:
            shutil.rmtree(p, ignore_errors=True)


def run_synthea(patient_count: int = 10) -> bool:
    """Run Synthea to generate patient data. Returns True on success."""
    if not RUN_SYNTHEA.exists():
        print(f"run_synthea not found at {RUN_SYNTHEA}", file=sys.stderr)
        return False
    cmd = [str(RUN_SYNTHEA), "-p", str(patient_count)]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=SCRIPT_DIR)
    return result.returncode == 0


def is_patient_bundle(path: Path) -> bool:
    name = path.name
    if name.startswith(SKIP_PREFIXES):
        return False
    if not name.endswith(".json"):
        return False
    return True


def _format_instant(dt: datetime) -> str:
    """Format datetime as FHIR instant (±HH:MM)."""
    s = dt.isoformat()
    if s[-3] != ":" and len(s) >= 5 and s[-5] in "+-":
        s = s[:-2] + ":" + s[-2:]
    return s


def parse_iso(s: str) -> datetime:
    # Handle optional trailing Z and ±HH:MM
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


SYNTHEA_NPI_SYSTEM = "http://hl7.org/fhir/sid/us-npi"
SYNTHEA_ORG_SYSTEM = "https://github.com/synthetichealth/synthea"


def _practitioner_ref_from_npi(npi: str) -> str:
    return f"Practitioner?identifier={SYNTHEA_NPI_SYSTEM}|{npi}"


def _parse_org_id_from_reference(ref: Optional[str]) -> Optional[str]:
    """Extract organization identifier value from Encounter.serviceProvider reference."""
    if not ref or "|" not in ref:
        return None
    return ref.strip().split("|")[-1].strip()


def _load_synthea_practitioner_data(fhir_dir: Path) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Load from Synthea practitionerInformation*.json:
    - all_refs: all practitioner references (for global fallback).
    - practitioners_by_org: org_identifier_value -> list of practitioner refs (from PractitionerRole).
    So when an encounter has no practitioner, we can prefer practitioners at that encounter's organization.
    """
    all_refs = []
    practitioners_by_org: Dict[str, List[str]] = {}
    for path in fhir_dir.glob("practitionerInformation*.json"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for ent in data.get("entry") or []:
                resource = ent.get("resource")
                if not resource:
                    continue
                rt = resource.get("resourceType")
                if rt == "Practitioner":
                    for ident in resource.get("identifier") or []:
                        if isinstance(ident, dict) and ident.get("system") == SYNTHEA_NPI_SYSTEM:
                            npi = ident.get("value")
                            if npi:
                                all_refs.append(_practitioner_ref_from_npi(npi))
                            break
                elif rt == "PractitionerRole":
                    npi = None
                    org_id = None
                    pract = resource.get("practitioner")
                    if isinstance(pract, dict) and isinstance(pract.get("identifier"), dict):
                        id_ = pract["identifier"]
                        if id_.get("system") == SYNTHEA_NPI_SYSTEM:
                            npi = id_.get("value")
                    org = resource.get("organization")
                    if isinstance(org, dict) and isinstance(org.get("identifier"), dict):
                        id_ = org["identifier"]
                        if id_.get("system") == SYNTHEA_ORG_SYSTEM:
                            org_id = id_.get("value")
                    if npi and org_id:
                        ref = _practitioner_ref_from_npi(npi)
                        if org_id not in practitioners_by_org:
                            practitioners_by_org[org_id] = []
                        if ref not in practitioners_by_org[org_id]:
                            practitioners_by_org[org_id].append(ref)
        except (json.JSONDecodeError, OSError):
            continue
    all_refs = list(dict.fromkeys(all_refs))
    return all_refs, practitioners_by_org


def _encounter_qualifies_for_appointment(encounter_entry: dict, rng) -> bool:
    """
    Option 1: Only certain encounter classes (ambulatory/outpatient; skip emergency/inpatient/urgent).
    Option 2: Probability filter (e.g. 40% of qualifying encounters get appointments).
    """
    resource = encounter_entry.get("resource") or encounter_entry
    enc_class = resource.get("class") or {}
    code = enc_class.get("code") if isinstance(enc_class, dict) else None
    if code not in ENCOUNTER_CLASS_CODES_FOR_APPOINTMENTS:
        return False
    if rng.random() >= APPOINTMENT_PROBABILITY:
        return False
    return True


def _synthetic_practitioner_ref(rng) -> str:
    """Fallback when no Synthea practitioner data exists (random NPI-style)."""
    npi = str(rng.randint(1000000000, 9999999999))
    return f"Practitioner?identifier=http://hl7.org/fhir/sid/us-npi|{npi}"


def _build_participants_and_service_type(
    encounter_entry: dict,
    patient_ref: str,
    rng,
    synthea_practitioner_refs: Optional[List[str]] = None,
    practitioners_by_org: Optional[Dict[str, List[str]]] = None,
) -> tuple:
    """Extract participant refs and serviceType from encounter. Returns (participants_list, service_type or None)."""
    resource = encounter_entry.get("resource") or encounter_entry
    practitioner_ref = None
    location_ref = None
    for p in resource.get("participant") or []:
        ind = p.get("individual") or {}
        ref = ind.get("reference") if isinstance(ind, dict) else None
        if ref and "Practitioner" in ref:
            practitioner_ref = ref
            break
    for loc in resource.get("location") or []:
        loc_ref = (loc.get("location") or {}).get("reference") if isinstance(loc.get("location"), dict) else None
        if loc_ref:
            location_ref = loc_ref
            break
    if not practitioner_ref and synthea_practitioner_refs:
        # Prefer practitioners at this encounter's organization (realistic: same hospital/clinic)
        sp = resource.get("serviceProvider")
        ref = sp.get("reference") if isinstance(sp, dict) else (sp if isinstance(sp, str) else None)
        org_id = _parse_org_id_from_reference(ref)
        candidates = None
        if org_id and practitioners_by_org and org_id in practitioners_by_org:
            candidates = practitioners_by_org[org_id]
        if not candidates:
            candidates = synthea_practitioner_refs
        practitioner_ref = rng.choice(candidates)
    elif not practitioner_ref:
        practitioner_ref = _synthetic_practitioner_ref(rng)
    participants = [
        {"actor": {"reference": patient_ref}, "required": "required", "status": "accepted"},
        {"actor": {"reference": practitioner_ref}, "required": "required", "status": "accepted"},
    ]
    if location_ref:
        participants.append({"actor": {"reference": location_ref}, "required": "optional", "status": "accepted"})
    encounter_types = resource.get("type") or []
    service_type = encounter_types[0] if isinstance(encounter_types, list) and encounter_types else None
    if not isinstance(service_type, dict):
        service_type = None
    return participants, service_type


def _one_appointment(
    status: str,
    start: datetime,
    end: datetime,
    participants: list,
    service_type: Optional[dict],
    encounter_full_url: str,
    rng,
) -> dict:
    """Build a single Appointment resource with given status and period."""
    appt_id = str(uuid.uuid4())
    start_s = _format_instant(start)
    end_s = _format_instant(end)
    for key, s in (("start", start_s), ("end", end_s)):
        if len(s) >= 5 and s[-5] in "+-" and s[-3] != ":":
            s = s[:-2] + ":" + s[-2:]
        if key == "start":
            start_s = s
        else:
            end_s = s
    appointment = {
        "resourceType": "Appointment",
        "id": appt_id,
        "meta": {"profile": ["http://hl7.org/fhir/us/core/StructureDefinition/us-core-appointment"]},
        "identifier": [{"system": "https://github.com/synthetichealth/synthea", "value": appt_id}],
        "status": status,
        "serviceType": [service_type] if service_type else [],
        "start": start_s,
        "end": end_s,
        "minutesDuration": max(1, int((end - start).total_seconds() // 60)),
        "participant": participants,
        "basedOn": [{"reference": encounter_full_url}],
    }
    if status in ("cancelled", "noshow"):
        appointment["cancellationReason"] = rng.choice(CANCEL_REASONS)
    return appointment


def encounter_appointments(
    encounter: dict,
    patient_ref: str,
    encounter_full_url: str,
    rng,
    synthea_practitioner_refs: Optional[List[str]] = None,
    practitioners_by_org: Optional[Dict[str, List[str]]] = None,
) -> List[dict]:
    """
    Build 2 or 3 Appointments per Encounter (dynamic), modeling BOOKED → ARRIVED → FULFILLED.
    All link to the same encounter via basedOn (scheduling → visit linkage).
    """
    resource = encounter.get("resource") or encounter
    if resource.get("resourceType") != "Encounter":
        return []
    period = resource.get("period") or {}
    start_str = period.get("start")
    end_str = period.get("end")
    if not start_str or not end_str:
        return []
    try:
        enc_start = parse_iso(start_str)
        enc_end = parse_iso(end_str)
    except Exception:
        return []
    participants, service_type = _build_participants_and_service_type(
        encounter, patient_ref, rng, synthea_practitioner_refs, practitioners_by_org
    )
    # Scheduled slot (before encounter)
    offset_min = rng.choice(APPT_OFFSET_MINUTES)
    slot_start = enc_start - timedelta(minutes=offset_min)
    duration = rng.choice(DURATION_MINUTES)
    slot_end = slot_start + timedelta(minutes=duration)
    if slot_end > enc_end:
        slot_end = enc_end
    if slot_end <= slot_start:
        slot_end = slot_start + timedelta(minutes=15)
    # Dynamic: 2 or 3 appointments per encounter
    num_appts = rng.choice([2, 3])
    if num_appts == 2:
        # BOOKED (scheduled slot) → FULFILLED (encounter)
        statuses_and_times = [
            ("booked", slot_start, slot_end),
            ("fulfilled", enc_start, enc_end),
        ]
    else:
        # BOOKED → ARRIVED → FULFILLED
        arrived_end = enc_start + timedelta(minutes=min(15, max(1, int((enc_end - enc_start).total_seconds() // 60) // 2)))
        statuses_and_times = [
            ("booked", slot_start, slot_end),
            ("arrived", enc_start, arrived_end),
            ("fulfilled", enc_start, enc_end),
        ]
    out = []
    for status, start, end in statuses_and_times:
        out.append(_one_appointment(status, start, end, participants, service_type, encounter_full_url, rng))
    return out


def add_appointments_to_bundle(
    bundle_path: Path,
    rng,
    synthea_practitioner_refs: Optional[List[str]] = None,
    practitioners_by_org: Optional[Dict[str, List[str]]] = None,
    max_appointments_for_patient: Optional[int] = None,
) -> int:
    """Add Appointment entries to one bundle. Returns number of appointments added."""
    if max_appointments_for_patient is None:
        max_appointments_for_patient = rng.randint(MIN_APPOINTMENTS_PER_PATIENT, MAX_APPOINTMENTS_PER_PATIENT)
    with open(bundle_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    entries = data.get("entry")
    if not entries:
        return 0
    patient_ref = None
    encounter_entries = []
    for ent in entries:
        full_url = ent.get("fullUrl") or ""
        resource = ent.get("resource") or {}
        rt = resource.get("resourceType")
        if rt == "Patient":
            patient_ref = full_url
        elif rt == "Encounter":
            encounter_entries.append((full_url, ent))
    if not patient_ref:
        return 0
    new_entries = []
    for full_url, ent in encounter_entries:
        if len(new_entries) >= max_appointments_for_patient:
            break
        if not _encounter_qualifies_for_appointment(ent, rng):
            continue
        appts = encounter_appointments(
            ent, patient_ref, full_url, rng, synthea_practitioner_refs, practitioners_by_org
        )
        if len(new_entries) + len(appts) > max_appointments_for_patient:
            continue
        for appt in appts:
            appt_id = appt["id"]
            new_entries.append({
                "fullUrl": f"urn:uuid:{appt_id}",
                "resource": appt,
                "request": {"method": "POST", "url": "Appointment"},
            })
    # Ensure at least one appointment per patient (use first AMB encounter if none from probability)
    if not new_entries and encounter_entries:
        for full_url, ent in encounter_entries:
            res = ent.get("resource") or ent
            cls_code = (res.get("class") or {}).get("code") if isinstance(res.get("class"), dict) else None
            if cls_code not in ENCOUNTER_CLASS_CODES_FOR_APPOINTMENTS:
                continue
            appts = encounter_appointments(
                ent, patient_ref, full_url, rng, synthea_practitioner_refs, practitioners_by_org
            )
            if appts:
                appt = appts[0]
                new_entries.append({
                    "fullUrl": f"urn:uuid:{appt['id']}",
                    "resource": appt,
                    "request": {"method": "POST", "url": "Appointment"},
                })
                break
        # If no AMB encounter exists, use first encounter of any class
        if not new_entries and encounter_entries:
            full_url, ent = encounter_entries[0]
            appts = encounter_appointments(
                ent, patient_ref, full_url, rng, synthea_practitioner_refs, practitioners_by_org
            )
            if appts:
                appt = appts[0]
                new_entries.append({
                    "fullUrl": f"urn:uuid:{appt['id']}",
                    "resource": appt,
                    "request": {"method": "POST", "url": "Appointment"},
                })
    if not new_entries:
        return 0
    # Insert appointments after the first Encounter (so Patient and early resources stay first)
    first_enc_idx = next((i for i, e in enumerate(entries) if (e.get("resource") or {}).get("resourceType") == "Encounter"), len(entries))
    for i, new in enumerate(new_entries):
        entries.insert(first_enc_idx + i, new)
    with open(bundle_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return len(new_entries)


def main():
    import random
    argv = sys.argv[1:]
    run_synthea_first = "--no-synthea" not in argv
    patient_count = 10
    max_bundle_size_mb = None  # None = don't delete any bundles
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "-p" and i + 1 < len(argv):
            patient_count = int(argv[i + 1])
            i += 2
            continue
        if arg.startswith("-p") and len(arg) > 2:
            patient_count = int(arg[2:])
        elif arg.startswith("--patients="):
            patient_count = int(arg.split("=", 1)[1])
        elif arg == "--max-bundle-size" and i + 1 < len(argv):
            max_bundle_size_mb = float(argv[i + 1])
            i += 2
            continue
        elif arg.startswith("--max-bundle-size="):
            max_bundle_size_mb = float(arg.split("=", 1)[1])
        i += 1
    if run_synthea_first:
        _clear_fhir_output(OUTPUT_FHIR)
        if not run_synthea(patient_count):
            sys.exit(1)
    if not OUTPUT_FHIR.is_dir():
        print(f"Output directory not found: {OUTPUT_FHIR}", file=sys.stderr)
        sys.exit(1)
    synthea_practitioner_refs, practitioners_by_org = _load_synthea_practitioner_data(OUTPUT_FHIR)
    if synthea_practitioner_refs:
        n_orgs = len(practitioners_by_org)
        print(f"Loaded {len(synthea_practitioner_refs)} practitioner(s) across {n_orgs} organization(s) from Synthea data.")
    total_added = 0
    files_processed = 0
    for path in sorted(OUTPUT_FHIR.glob("*.json")):
        if not is_patient_bundle(path):
            continue
        # Seed RNG from filename so output is stable per file but varies across files
        seed = hash(path.name) % (2**32)
        rng = random.Random(seed)
        n = add_appointments_to_bundle(path, rng, synthea_practitioner_refs, practitioners_by_org)
        if n:
            total_added += n
            files_processed += 1
            print(f"  {path.name}: added {n} Appointment(s)")
    print(f"Done. Added {total_added} Appointment(s) across {files_processed} patient bundle(s).")

    # Delete patient bundles over --max-bundle-size MB (only if flag was given)
    if max_bundle_size_mb is not None:
        limit_bytes = int(max_bundle_size_mb * 1024 * 1024)
        removed = 0
        for path in list(OUTPUT_FHIR.glob("*.json")):
            if not is_patient_bundle(path):
                continue
            try:
                if path.stat().st_size > limit_bytes:
                    path.unlink()
                    removed += 1
                    print(f"  Removed {path.name} (>{max_bundle_size_mb} MB)")
            except OSError:
                pass
        if removed:
            print(f"Removed {removed} patient bundle(s) exceeding {max_bundle_size_mb} MB.")


if __name__ == "__main__":
    main()