"""
DeskPilot AI - Synthetic Dataset Generator
==============================================
Generates realistic IT helpdesk ticket data for a global enterprise service desk
(the scenario: employees currently browse a large service catalog to raise
Incidents / Service Requests / Change Requests (RFCs) / Admin requests -
this dataset trains a model to route free-text requests directly instead).

Design notes (why synthetic, and why this way):
- No proprietary/scraped data is used - this is fully synthetic and safe to publish.
- Text is generated from templates + slot-filling + linguistic noise (typos,
  abbreviations, informal phrasing, punctuation variance) so the classification
  task is non-trivial - a pure keyword-match system would NOT solve this well,
  which is the point of using ML here.
- Priority is NOT a deterministic function of category alone. It's influenced by
  category base-rate + requester seniority + department criticality + random
  business noise. This mirrors reality (a VP's "software install" can outrank an
  associate's "P3 incident") and is *why* the model fuses text + tabular features
  for the priority head instead of predicting priority from text alone.
"""
import random
import uuid

import pandas as pd
from faker import Faker

import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

fake = Faker()
random.seed(42)
Faker.seed(42)

# ---------------------------------------------------------------------------
# 1. Catalog taxonomy: 4 request types -> 19 catalog categories
# ---------------------------------------------------------------------------
CATALOG = {
    "Incident": {
        "Network/VPN Connectivity Issue": {
            "priority_weights": {"P1": 0.35, "P2": 0.40, "P3": 0.20, "P4": 0.05},
            "templates": [
                "{greet}my VPN keeps disconnecting every {time_unit}, can't {work_verb}",
                "unable to connect to office VPN since {time_ref}, {impact}",
                "vpn client shows error {err_code} on {os}, blocking access to {resource}",
                "internet/vpn down at {location} floor, whole team affected",
                "cant access {resource} over vpn, connection times out",
                "vpn keeps dropping while on client call, very urgent",
                "getting 'unable to establish secure connection' on vpn client",
            ],
        },
        "Application Crash or Error": {
            "priority_weights": {"P1": 0.20, "P2": 0.45, "P3": 0.30, "P4": 0.05},
            "templates": [
                "{app} crashes every time i {work_verb}, error code {err_code}",
                "{app} is throwing an exception on startup since {time_ref}",
                "getting a white screen / blank page in {app}, cleared cache already",
                "{app} freezes when i try to {work_verb}, need this fixed {urgency_phrase}",
                "unable to open {app}, keeps saying 'application not responding'",
                "{app} lost all my unsaved work after crashing, please help",
            ],
        },
        "Email/Outlook Issue": {
            "priority_weights": {"P1": 0.15, "P2": 0.40, "P3": 0.35, "P4": 0.10},
            "templates": [
                "not receiving emails since {time_ref}, checked spam already",
                "outlook keeps freezing / not syncing on {os}",
                "unable to send emails, getting bounce back error {err_code}",
                "outlook calendar not showing meeting invites from {dept} team",
                "mailbox full error, need cleanup or quota increase {urgency_phrase}",
                "shared mailbox for {dept} not accessible since this morning",
            ],
        },
        "Printer Not Working": {
            "priority_weights": {"P1": 0.02, "P2": 0.13, "P3": 0.45, "P4": 0.40},
            "templates": [
                "printer on {location} floor showing offline, tried restart",
                "print jobs stuck in queue for {dept} printer",
                "printer paper jam error even after clearing tray",
                "cant print from {app}, other apps print fine",
                "network printer not detected on {os}",
                "{greet}toner/ink warning on {location} floor printer, needs replacement",
                "scan to email not working from {location} floor printer",
            ],
        },
        "Login/Authentication Failure": {
            "priority_weights": {"P1": 0.40, "P2": 0.40, "P3": 0.18, "P4": 0.02},
            "templates": [
                "locked out of my account after {n} failed attempts, {urgency_phrase}",
                "sso login failing with error {err_code} across all apps",
                "mfa/authenticator code not accepted, cant get into {resource}",
                "password reset link expired, still cant log in",
                "account shows 'disabled' when trying to sign in to {resource}",
            ],
        },
        "Hardware Malfunction": {
            "priority_weights": {"P1": 0.10, "P2": 0.35, "P3": 0.40, "P4": 0.15},
            "templates": [
                "laptop battery draining within {n} mins, needs replacement",
                "laptop screen flickering / has dead pixels since {time_ref}",
                "keyboard keys not responding on my {os} laptop",
                "laptop overheating and shutting down randomly during {work_verb}",
                "docking station not detecting external monitor",
                "laptop wont boot past {os} logo screen",
            ],
        },
    },
    "Service Request": {
        "Software Installation Request": {
            "priority_weights": {"P1": 0.02, "P2": 0.10, "P3": 0.38, "P4": 0.50},
            "templates": [
                "need {app} installed on my laptop for {dept} work",
                "please install {app}, required for {work_verb} on current project",
                "can someone set up {app} on my machine {urgency_phrase}",
                "requesting installation of {app} - approved by {manager}",
                "new joiner needs {app} + standard {dept} toolset installed",
            ],
        },
        "Software License Request": {
            "priority_weights": {"P1": 0.01, "P2": 0.09, "P3": 0.40, "P4": 0.50},
            "templates": [
                "need a license for {app}, trial about to expire",
                "requesting additional {app} seat for new {dept} hire",
                "our team's {app} license renewal is due, please process",
                "need enterprise license upgrade for {app} for {n} users",
            ],
        },
        "New Hardware/Peripheral Request": {
            "priority_weights": {"P1": 0.01, "P2": 0.07, "P3": 0.37, "P4": 0.55},
            "templates": [
                "requesting a new laptop, current one is {n} years old and slow",
                "need an external monitor for wfh setup, approved by {manager}",
                "requesting a headset/webcam for client calls",
                "need a docking station for the {dept} team's new hires",
                "requesting a laptop bag/mouse replacement, old one damaged",
            ],
        },
        "Access/Permission Request": {
            "priority_weights": {"P1": 0.08, "P2": 0.30, "P3": 0.42, "P4": 0.20},
            "templates": [
                "need read/write access to {resource} for {dept} project",
                "requesting admin rights on my machine for {work_verb}",
                "please grant access to {resource} repo, approved by {manager}",
                "need access removed for a former team member on {resource}",
                "requesting elevated access to {resource} to debug prod issue {urgency_phrase}",
            ],
        },
        "Password Reset": {
            "priority_weights": {"P1": 0.30, "P2": 0.42, "P3": 0.23, "P4": 0.05},
            "templates": [
                "forgot my password, need a reset {urgency_phrase}",
                "self-service password reset isnt working, need manual reset",
                "need password reset for shared {dept} service account",
                "locked out after password expired, cant reset it myself",
                "{greet}password reset email never arrived, checked spam {urgency_phrase}",
                "need my {resource} password reset, self reset option is greyed out",
                "reset password request for new {dept} joiner's account",
                "password reset otp not arriving on registered mobile number",
            ],
        },
        "Distribution List / Mailbox Request": {
            "priority_weights": {"P1": 0.01, "P2": 0.09, "P3": 0.40, "P4": 0.50},
            "templates": [
                "please add me to the {dept} distribution list",
                "need a new shared mailbox created for {dept} team",
                "requesting removal of old members from {dept} DL",
                "need a new distribution list set up for the {dept} project",
                "requesting {manager} be added as owner of {dept} DL",
                "please remove me from the old {dept} mailing list",
            ],
        },
        "VPN Access Setup": {
            "priority_weights": {"P1": 0.05, "P2": 0.25, "P3": 0.45, "P4": 0.25},
            "templates": [
                "need vpn access configured for new remote hire in {dept}",
                "requesting vpn profile setup on my new laptop",
                "need site-to-site vpn access to {resource} for the project",
                "requesting vpn access renewal, mine expired {time_ref}",
                "please enable vpn for contractor working with {dept} team",
                "need vpn client + certificate installed on my {os} machine",
            ],
        },
    },
    "Change Request": {
        "Server Configuration Change": {
            "priority_weights": {"P1": 0.15, "P2": 0.45, "P3": 0.30, "P4": 0.10},
            "templates": [
                "raising RFC to increase memory allocation on {resource}",
                "need config change on {resource} to fix recurring timeout",
                "RFC: update server timezone/locale settings on {resource}",
                "requesting scheduled restart of {resource} this weekend",
                "RFC to patch OS on {resource}, maintenance window needed",
                "raising change request to resize compute for {resource} before {time_ref}",
                "RFC to rotate credentials/certificates on {resource} {urgency_phrase}",
                "requesting config rollback on {resource} after last night's change",
            ],
        },
        "Firewall/Network Change": {
            "priority_weights": {"P1": 0.15, "P2": 0.40, "P3": 0.35, "P4": 0.10},
            "templates": [
                "need firewall rule opened for port {n} to {resource}",
                "RFC to whitelist new vendor IP for {resource} integration",
                "requesting network change to allow {app} traffic through proxy",
                "raising change request to update VPN split-tunnel rules",
            ],
        },
        "Database Schema/Deployment Change": {
            "priority_weights": {"P1": 0.20, "P2": 0.45, "P3": 0.28, "P4": 0.07},
            "templates": [
                "RFC to add new index on {resource} table for perf fix",
                "requesting approval to run migration script on {resource} db",
                "raising change for schema update on {resource}, tested in staging",
                "need RFC approved for db failover test on {resource}",
                "requesting approval to add new column to {resource} schema {urgency_phrase}",
                "RFC to run data backfill script on {resource} this weekend",
                "need change approval for db version upgrade on {resource}",
            ],
        },
        "Application Deployment/Release": {
            "priority_weights": {"P1": 0.18, "P2": 0.42, "P3": 0.30, "P4": 0.10},
            "templates": [
                "RFC for production release of {app} v{n}.0 this sprint",
                "requesting deployment window for {app} hotfix {urgency_phrase}",
                "raising change request to roll back {app} deployment",
                "need approval to deploy {app} config change to prod",
            ],
        },
    },
    "Admin": {
        "Employee Onboarding Setup": {
            "priority_weights": {"P1": 0.03, "P2": 0.20, "P3": 0.47, "P4": 0.30},
            "templates": [
                "new hire joining {dept} on {time_ref}, needs full IT setup",
                "please prepare laptop + accounts for new {dept} joiner",
                "onboarding checklist pending for new associate in {dept}",
            ],
        },
        "Employee Offboarding/Asset Return": {
            "priority_weights": {"P1": 0.10, "P2": 0.35, "P3": 0.40, "P4": 0.15},
            "templates": [
                "employee last working day is {time_ref}, please disable access",
                "need asset return + access revoke processed for {dept} employee",
                "offboarding: revoke {resource} access for exiting team member {urgency_phrase}",
            ],
        },
    },
}

APPS = ["Adobe Photoshop", "VS Code", "Slack", "Zoom", "Jira", "MS Teams", "Excel",
        "Tableau", "Figma", "Postman", "Docker Desktop", "IntelliJ IDEA", "SAP GUI",
        "Salesforce", "PowerBI", "Git client", "Anaconda", "Outlook", "WinSCP", "Putty"]
DEPTS = ["Finance", "HR", "Engineering", "Data Science", "Marketing", "Sales",
         "Legal", "Procurement", "Customer Support", "DevOps", "QA", "Design"]
RESOURCES = ["shared drive", "Jira project", "prod database", "GitHub repo",
             "internal wiki", "billing system", "CRM", "staging server",
             "client SFTP", "analytics dashboard", "build pipeline"]
OS_LIST = ["Windows 11", "Windows 10", "macOS", "Linux"]
ERR_CODES = ["0x8007045", "500", "403", "DNS_PROBE_FAILED", "TLS_ERROR", "0x80070005", "timeout-504"]
WORK_VERBS = ["run a report", "join the client call", "push code", "open a file",
              "export data", "start my shift", "submit the deliverable"]
URGENCY = ["asap", "before EOD", "urgently, blocking a client deliverable", "at the earliest",
           "today if possible", "", "", ""]
TIME_REF = ["yesterday", "this morning", "since last night", "after the last update",
            "for the past 2 days", "since the reboot"]
GREETS = ["hi team, ", "hello, ", "hi, ", "", "", ""]
IMPACT = ["blocking my work", "cant join standup", "missing deadline", "team is idle"]

ROLES = ["Associate", "Senior Associate", "Team Lead", "Manager", "Senior Manager", "Director", "VP"]
ROLE_WEIGHT = {"Associate": 0, "Senior Associate": 1, "Team Lead": 2, "Manager": 3,
               "Senior Manager": 4, "Director": 5, "VP": 6}

TYPO_MAP = {"the": "teh", "please": "plz", "cannot": "cant", "you": "u",
            "because": "bcoz", "issue": "isue", "receive": "recieve"}


def add_noise(text):
    words = text.split()
    if random.random() < 0.15 and len(words) > 3:
        i = random.randrange(len(words))
        w = words[i].lower().strip(".,")
        if w in TYPO_MAP:
            words[i] = TYPO_MAP[w]
    text = " ".join(words)
    if random.random() < 0.5 and text:
        text = text[0].upper() + text[1:]
    if random.random() < 0.3:
        text = text.rstrip(".")
    return text


def fill_template(tpl):
    return add_noise(tpl.format(
        app=random.choice(APPS), dept=random.choice(DEPTS), resource=random.choice(RESOURCES),
        os=random.choice(OS_LIST), err_code=random.choice(ERR_CODES),
        work_verb=random.choice(WORK_VERBS), urgency_phrase=random.choice(URGENCY),
        time_ref=random.choice(TIME_REF), greet=random.choice(GREETS), impact=random.choice(IMPACT),
        time_unit=random.choice(["5 minutes", "10 mins", "hour"]),
        n=random.randint(1, 48), location=random.choice(["3rd", "5th", "ground", "7th"]),
        manager=fake.first_name(),
    ))


SHARPEN_POWER = 2.3  # sharpens category base-rate so there's a learnable dominant signal,
                      # while keeping the task genuinely probabilistic (not deterministic)


def sample_priority(weights, role, dept):
    raw = [weights["P1"], weights["P2"], weights["P3"], weights["P4"]]
    sharp = [w ** SHARPEN_POWER for w in raw]
    s = sum(sharp)
    p1, p2, p3, p4 = [w / s for w in sharp]

    pull = ROLE_WEIGHT[role] / 12.0
    if dept in ("Engineering", "DevOps", "Data Science"):
        pull += 0.05
    p1 += pull * 0.5
    p2 += pull * 0.3
    p3 = max(p3 - pull * 0.5, 0.01)
    p4 = max(p4 - pull * 0.3, 0.01)
    total = p1 + p2 + p3 + p4
    probs = [p1 / total, p2 / total, p3 / total, p4 / total]
    return random.choices(["P1-Critical", "P2-High", "P3-Medium", "P4-Low"], weights=probs)[0]


# Category pairs that are genuinely easy for humans to mis-file in a real helpdesk
# (used to inject realistic label noise below - real annotators are not perfect,
# and pretending otherwise would make the eventual 100%-accuracy result meaningless).
CONFUSABLE_PAIRS = {
    "Software Installation Request": "Software License Request",
    "Software License Request": "Software Installation Request",
    "Application Crash or Error": "Hardware Malfunction",
    "Hardware Malfunction": "Application Crash or Error",
    "Access/Permission Request": "VPN Access Setup",
    "VPN Access Setup": "Network/VPN Connectivity Issue",
    "Network/VPN Connectivity Issue": "VPN Access Setup",
    "Server Configuration Change": "Database Schema/Deployment Change",
    "Database Schema/Deployment Change": "Server Configuration Change",
    "Employee Onboarding Setup": "New Hardware/Peripheral Request",
    "Distribution List / Mailbox Request": "Email/Outlook Issue",
}
LABEL_NOISE_RATE = 0.045  # ~4-5% mislabeled - realistic for human-tagged ticket data


def generate(n_rows=7000):
    rows = []
    cats = [(rt, cat) for rt, cs in CATALOG.items() for cat in cs]
    for _ in range(n_rows):
        req_type, category = random.choice(cats)
        spec = CATALOG[req_type][category]
        text = fill_template(random.choice(spec["templates"]))
        role = random.choices(ROLES, weights=[30, 22, 18, 14, 9, 5, 2])[0]
        dept = random.choice(DEPTS)
        priority = sample_priority(spec["priority_weights"], role, dept)
        ts = fake.date_time_between(start_date="-18M", end_date="now")
        rows.append({
            "ticket_id": f"TCK-{uuid.uuid4().hex[:8].upper()}",
            "query_text": text,
            "request_type": req_type,
            "category": category,
            "priority": priority,
            "requester_role": role,
            "department": dept,
            "submitted_at": ts.isoformat(timespec="seconds"),
        })
    df = pd.DataFrame(rows)
    # De-dupe on exact query text only (not category): prevents identical text ending up
    # with two different labels in the dataset, and prevents train/test leakage later
    # when we split - a data-hygiene step worth calling out explicitly in interviews.
    df = df.drop_duplicates(subset=["query_text"]).reset_index(drop=True)

    # Inject realistic label noise: real helpdesk agents mis-file tickets into a
    # neighboring/confusable category some of the time. We simulate that here so the
    # eventual model evaluation is honest (a classifier trained on templated text with
    # ZERO label noise would hit ~100% accuracy, which is a red flag, not a strength).
    n_noisy = 0
    for i in range(len(df)):
        true_cat = df.at[i, "category"]
        if true_cat in CONFUSABLE_PAIRS and random.random() < LABEL_NOISE_RATE:
            df.at[i, "category"] = CONFUSABLE_PAIRS[true_cat]
            n_noisy += 1
    print(f"Injected label noise into {n_noisy} rows ({n_noisy/len(df):.1%}) to simulate real annotator error")
    return df


if __name__ == "__main__":
    df = generate(26000)
    out_path = os.path.join(PROJECT_ROOT, "data", "v1", "tickets.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} rows")
    print("\n--- category distribution ---")
    print(df["category"].value_counts())
    print("\n--- priority distribution ---")
    print(df["priority"].value_counts(normalize=True).round(3))
    print("\nSample rows:")
    print(df.sample(5, random_state=1)[["query_text", "category", "priority"]].to_string(index=False))
