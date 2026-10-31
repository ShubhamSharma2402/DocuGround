"""Script to programmatically generate the evaluation PDF and 50 labeled QA pairs in JSONL."""

from __future__ import annotations

import json
from pathlib import Path
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import fitz  # PyMuPDF

DATASET_DIR = Path(__file__).parent
PDF_PATH = DATASET_DIR / "docuground_sample_report.pdf"
QA_JSONL_PATH = DATASET_DIR / "qa_pairs.jsonl"


PAGES_CONTENT = [
    # Page 1: Overview & Executive Summary
    (
        "Novasphere Technologies Inc. — 2024 Annual Review\n\n"
        "Executive Summary:\n"
        "Novasphere Technologies Inc. (NYSE: NVST) is a global provider of enterprise cloud orchestration, "
        "autonomous infrastructure management, and AI data systems. Founded in 2018 by Dr. Aris Thorne and Elena Rostova, "
        "the company is headquartered in Seattle, Washington, with primary international regional hubs in Zurich and Singapore.\n\n"
        "Fiscal Year 2024 Highlights:\n"
        "Novasphere delivered exceptional operational performance in fiscal year 2024. Total consolidated revenue reached "
        "$428.5 million, representing a 34.2% year-over-year growth from $319.3 million in fiscal year 2023. Gross margin "
        "expanded by 210 basis points to 78.4%. Free cash flow reached $94.2 million, up from $61.8 million in 2023.\n\n"
        "Our customer base expanded to 3,420 enterprise customers, with 482 customers contributing over $250,000 in Annual "
        "Recurring Revenue (ARR). The Net Revenue Retention (NRR) rate closed the year at 124%, underscoring strong product-market fit."
    ),

    # Page 2: Product Architecture & Platforms
    (
        "Product Portfolio and Core Technologies\n\n"
        "Novasphere operates three flagship platforms built upon a unified distributed data backbone:\n\n"
        "1. NovaOrchestrator:\n"
        "An autonomous Kubernetes control plane that optimizes multi-cloud workload placement across AWS, Azure, and Google Cloud. "
        "In 2024, NovaOrchestrator managed over 14.2 million container nodes daily, delivering an average cloud compute cost "
        "reduction of 28.5% for enterprise clients. NovaOrchestrator uses the proprietary Helios-4 scheduling engine.\n\n"
        "2. PulseMesh:\n"
        "A zero-trust service mesh engineered for microsecond-latency service-to-service communication. PulseMesh implements "
        "mTLS 1.3 by default and incorporates real-time anomaly detection, processing 4.8 trillion network packets per day.\n\n"
        "3. AstraStore:\n"
        "A high-throughput distributed vector database and object storage engine optimized for enterprise Retrieval-Augmented "
        "Generation (RAG) and model weights. AstraStore achieves sub-5-millisecond p99 query latency on 100-million-vector indexes."
    ),

    # Page 3: Financial Performance & Segment Revenue
    (
        "Financial Results and Revenue Disaggregation\n\n"
        "Consolidated Statement of Operations (in millions of USD, except share data):\n"
        "• Subscription Revenue: $381.4M (vs $278.1M in FY23)\n"
        "• Professional Services and Training: $47.1M (vs $41.2M in FY23)\n"
        "• Total Revenue: $428.5M (vs $319.3M in FY23)\n"
        "• Cost of Subscription: $61.0M\n"
        "• Cost of Services: $31.5M\n"
        "• Total Cost of Revenue: $92.5M\n"
        "• Gross Profit: $336.0M (Gross Margin: 78.4%)\n\n"
        "Operating Expenses:\n"
        "• Research and Development: $138.2M\n"
        "• Sales and Marketing: $118.6M\n"
        "• General and Administrative: $44.1M\n"
        "• Total Operating Expenses: $300.9M\n"
        "• Operating Income: $35.1M (Operating Margin: 8.2%)\n"
        "• GAAP Net Income: $28.6M ($0.34 per diluted share)\n"
        "• Non-GAAP Net Income: $68.4M ($0.81 per diluted share)"
    ),

    # Page 4: Geographic Performance & Customer Segments
    (
        "Geographic Breakdown and Market Segment Analysis\n\n"
        "Regional Performance:\n"
        "Revenue distribution across geographic theaters during fiscal 2024 was as follows:\n"
        "• North America (United States & Canada): $265.7 million (62.0% of total revenue)\n"
        "• EMEA (Europe, Middle East, Africa): $115.7 million (27.0% of total revenue)\n"
        "• APAC (Asia-Pacific): $47.1 million (11.0% of total revenue)\n\n"
        "APAC represented the fastest-growing territory with an annual growth rate of 51.4%, driven by rapid adoption in Japan, "
        "Australia, and Singapore. North America expanded by 29.8% and EMEA grew by 36.1%.\n\n"
        "Industry Verticals:\n"
        "Financial Services constituted our largest single vertical, generating 38% of total subscription revenue, followed by "
        "Healthcare and Life Sciences at 24%, Technology & Telecommunications at 21%, and Retail/E-commerce at 17%."
    ),

    # Page 5: Infrastructure & Engineering Architecture
    (
        "Infrastructure Architecture and Deployment Topology\n\n"
        "Novasphere's software-as-a-service infrastructure operates in 26 global cloud regions across three primary hyperscalers. "
        "Our control plane is partitioned into cells, with each cell supporting up to 50,000 active nodes.\n\n"
        "Architecture Components:\n"
        "• Gateway Layer: Custom Envoy proxy fleet with HTTP/3 and QUIC support.\n"
        "• Consensus Protocol: Modified Raft consensus with asynchronous snapshot replication (Raft-Nova v2).\n"
        "• Storage Tier: RocksDB storage engine for local fast state, backed by Amazon S3 and Google Cloud Storage.\n"
        "• Memory Footprint: Median agent footprint is 18 MB RAM with under 0.5% CPU utilization under peak load.\n\n"
        "Reliability and SLA:\n"
        "In 2024, our global platform achieved 99.994% system availability, exceeding our contractual service-level agreement (SLA) "
        "of 99.99%. Total unscheduled downtime across all 26 regions was 31.5 minutes for the entire year."
    ),

    # Page 6: Information Security & Data Governance
    (
        "Security, Privacy, and Compliance Framework\n\n"
        "Data Governance and Certifications:\n"
        "Novasphere maintains the following active compliance certifications:\n"
        "• SOC 2 Type II across Security, Availability, and Confidentiality trust principles.\n"
        "• ISO/IEC 27001:2022 and ISO/IEC 27701 (Privacy Information Management).\n"
        "• HIPAA compliance verified by external third-party auditor Schellman & Company.\n"
        "• FedRAMP Moderate In-Process designation achieved in Q3 2024, sponsored by the Department of Commerce.\n\n"
        "Encryption Standards:\n"
        "All customer data is encrypted in transit using TLS 1.3 with AES-256-GCM ciphers. Data at rest is encrypted using "
        "envelope encryption with AES-256 keys rotated automatically every 90 days. Customers may bring their own keys (BYOK) "
        "using AWS KMS, Azure Key Vault, or HashiCorp Vault. Zero cleartext customer tokens are stored in persistence layers."
    ),

    # Page 7: Research & Development and Patents
    (
        "Research, Development, and Intellectual Property\n\n"
        "R&D Strategy and Investments:\n"
        "During fiscal 2024, Novasphere invested $138.2 million in Research and Development, representing 32.3% of total revenue. "
        "Our engineering team numbered 412 full-time employees as of December 31, 2024, representing 47% of total company headcount.\n\n"
        "Intellectual Property Portfolio:\n"
        "Novasphere held 34 granted patents and 52 pending patent applications with the United States Patent and Trademark Office (USPTO). "
        "Key patents include:\n"
        "• US Patent 11,842,109: 'Autonomous predictive scaling in heterogeneous cloud container environments'\n"
        "• US Patent 11,928,443: 'Distributed sparse-dense index merging for low-latency similarity retrieval'\n"
        "• US Patent 12,014,882: 'Decentralized mutual authentication in zero-trust mesh topologies'\n\n"
        "In 2024, the team published 7 peer-reviewed papers at NeurIPS, OSDI, and SIGCOMM."
    ),

    # Page 8: Corporate Governance & Leadership
    (
        "Executive Leadership and Board of Directors\n\n"
        "Executive Management:\n"
        "• Dr. Aris Thorne — Chief Executive Officer and Co-Founder (former VP of Cloud Systems at Oracle)\n"
        "• Elena Rostova — Chief Technology Officer and Co-Founder (PhD in Distributed Computing from ETH Zurich)\n"
        "• Marcus Vance — Chief Financial Officer (former CFO at Datastream Corp)\n"
        "• Sarah Chen-Boucher — Chief Revenue Officer (joined in February 2023 from HashiCorp)\n"
        "• Tariq Al-Mansoor — Chief Legal Officer & Corporate Secretary\n\n"
        "Board of Directors:\n"
        "The board consists of 7 directors, including 5 independent members. Arthur Pendelton serves as independent Chairman "
        "of the Board. The Audit Committee is chaired by Kimberly Walsh (former Audit Partner at PwC)."
    ),

    # Page 9: Environmental, Social, and Governance (ESG)
    (
        "Sustainability, ESG, and Workplace Diversity\n\n"
        "Environmental Sustainability:\n"
        "Novasphere committed to achieving net-zero greenhouse gas emissions across Scope 1, 2, and 3 by 2030. In 2024:\n"
        "• 92% of our compute workloads were hosted in data centers powered by 100% renewable energy contracts.\n"
        "• NovaOrchestrator's carbon-aware routing algorithm helped customers avoid an estimated 42,000 metric tons of CO2e.\n"
        "• Corporate office facilities in Seattle and Zurich achieved LEED Gold certification.\n\n"
        "Human Capital & Diversity:\n"
        "Total global workforce stood at 876 employees across 14 countries. Women represented 36% of our global workforce and "
        "31% of technical engineering roles. Underrepresented minorities comprised 28% of our United States employee population."
    ),

    # Page 10: Risk Factors and Legal Proceedings
    (
        "Risk Factors and Forward-Looking Disclosures\n\n"
        "Key Enterprise Risk Factors:\n"
        "1. Cloud Provider Concentration:\n"
        "Substantially all of our production infrastructure is hosted on AWS, Azure, and Google Cloud. Any disruption or "
        "unilateral price change by these cloud providers could materially impact our gross margins and operating results.\n\n"
        "2. Cybersecurity and Service Interruption:\n"
        "Although we employ multi-layered security controls, our systems remain subject to sophisticated cyberattacks, distributed "
        "denial-of-service (DDoS) incidents, and zero-day vulnerabilities in open-source dependencies.\n\n"
        "3. International Sanctions and Export Controls:\n"
        "We are subject to United States Export Administration Regulations (EAR). Changes in semiconductor or AI export rules "
        "may restrict our ability to sell advanced orchestration solutions in specific international jurisdictions.\n\n"
        "Legal Proceedings:\n"
        "As of December 31, 2024, Novasphere was not party to any material pending litigation or regulatory enforcement actions."
    )
]


def create_sample_pdf() -> Path:
    """Generate the 10-page sample PDF."""
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()

    for page_num, text in enumerate(PAGES_CONTENT, 1):
        page = doc.new_page(width=612, height=792)  # Standard Letter size
        # Header banner
        page.draw_rect(fitz.Rect(50, 40, 562, 42), color=(0.2, 0.4, 0.8), fill=(0.2, 0.4, 0.8))
        page.insert_text(fitz.Point(50, 35), "Novasphere Technologies Inc. — Form 10-K Excerpt", fontsize=9, color=(0.3, 0.3, 0.3))
        page.insert_text(fitz.Point(520, 35), f"Page {page_num}", fontsize=9, color=(0.3, 0.3, 0.3))

        # Main text
        rect = fitz.Rect(50, 60, 562, 730)
        page.insert_textbox(rect, text, fontsize=11, fontname="helv", color=(0.1, 0.1, 0.1))

        # Footer
        page.insert_text(fitz.Point(220, 760), "CONFIDENTIAL & PROPRIETARY — NVST-2024", fontsize=8, color=(0.5, 0.5, 0.5))

    doc.save(str(PDF_PATH))
    doc.close()
    return PDF_PATH


QA_PAIRS = [
    # ── Answerable (35 items) ────────────────────────────────────────────────
    {
        "id": "q1",
        "question": "What was Novasphere's total consolidated revenue in fiscal year 2024?",
        "answerable": True,
        "expected_pages": [1, 3],
        "ground_truth": "Novasphere's total consolidated revenue in fiscal year 2024 was $428.5 million.",
        "key_terms": ["428.5", "million"]
    },
    {
        "id": "q2",
        "question": "Who are the co-founders of Novasphere Technologies and when was it founded?",
        "answerable": True,
        "expected_pages": [1, 8],
        "ground_truth": "Novasphere was founded in 2018 by Dr. Aris Thorne and Elena Rostova.",
        "key_terms": ["2018", "Aris Thorne", "Elena Rostova"]
    },
    {
        "id": "q3",
        "question": "Where is Novasphere headquartered and what are its primary international hubs?",
        "answerable": True,
        "expected_pages": [1],
        "ground_truth": "Novasphere is headquartered in Seattle, Washington, with international hubs in Zurich and Singapore.",
        "key_terms": ["Seattle", "Zurich", "Singapore"]
    },
    {
        "id": "q4",
        "question": "What was the gross margin and year-over-year revenue growth for fiscal 2024?",
        "answerable": True,
        "expected_pages": [1, 3],
        "ground_truth": "Gross margin was 78.4% and revenue grew 34.2% year-over-year.",
        "key_terms": ["78.4%", "34.2%"]
    },
    {
        "id": "q5",
        "question": "How much free cash flow did Novasphere generate in 2024 compared to 2023?",
        "answerable": True,
        "expected_pages": [1],
        "ground_truth": "Novasphere generated $94.2 million in free cash flow in 2024 compared to $61.8 million in 2023.",
        "key_terms": ["94.2", "61.8"]
    },
    {
        "id": "q6",
        "question": "What was the Net Revenue Retention (NRR) rate at the end of the year?",
        "answerable": True,
        "expected_pages": [1],
        "ground_truth": "The Net Revenue Retention (NRR) rate closed the year at 124%.",
        "key_terms": ["124%"]
    },
    {
        "id": "q7",
        "question": "How many total enterprise customers does Novasphere have, and how many contribute over $250k in ARR?",
        "answerable": True,
        "expected_pages": [1],
        "ground_truth": "Novasphere has 3,420 enterprise customers, with 482 contributing over $250,000 in ARR.",
        "key_terms": ["3,420", "482", "250,000"]
    },
    {
        "id": "q8",
        "question": "What are the three flagship products offered by Novasphere?",
        "answerable": True,
        "expected_pages": [2],
        "ground_truth": "The three flagship products are NovaOrchestrator, PulseMesh, and AstraStore.",
        "key_terms": ["NovaOrchestrator", "PulseMesh", "AstraStore"]
    },
    {
        "id": "q9",
        "question": "What scheduling engine is utilized by NovaOrchestrator?",
        "answerable": True,
        "expected_pages": [2],
        "ground_truth": "NovaOrchestrator uses the proprietary Helios-4 scheduling engine.",
        "key_terms": ["Helios-4"]
    },
    {
        "id": "q10",
        "question": "What average cloud compute cost reduction does NovaOrchestrator deliver?",
        "answerable": True,
        "expected_pages": [2],
        "ground_truth": "NovaOrchestrator delivers an average cloud compute cost reduction of 28.5%.",
        "key_terms": ["28.5%"]
    },
    {
        "id": "q11",
        "question": "What protocol version does PulseMesh implement by default for communication?",
        "answerable": True,
        "expected_pages": [2],
        "ground_truth": "PulseMesh implements mTLS 1.3 by default.",
        "key_terms": ["mTLS 1.3"]
    },
    {
        "id": "q12",
        "question": "What is AstraStore's p99 query latency on 100-million-vector indexes?",
        "answerable": True,
        "expected_pages": [2],
        "ground_truth": "AstraStore achieves sub-5-millisecond p99 query latency on 100-million-vector indexes.",
        "key_terms": ["sub-5-millisecond", "5"]
    },
    {
        "id": "q13",
        "question": "What was the breakdown between Subscription revenue and Professional Services in 2024?",
        "answerable": True,
        "expected_pages": [3],
        "ground_truth": "Subscription revenue was $381.4 million and Professional Services was $47.1 million.",
        "key_terms": ["381.4", "47.1"]
    },
    {
        "id": "q14",
        "question": "What were the Research and Development expenses in fiscal 2024?",
        "answerable": True,
        "expected_pages": [3, 7],
        "ground_truth": "Research and Development expenses were $138.2 million in fiscal 2024.",
        "key_terms": ["138.2"]
    },
    {
        "id": "q15",
        "question": "What was GAAP Net Income and GAAP diluted earnings per share for 2024?",
        "answerable": True,
        "expected_pages": [3],
        "ground_truth": "GAAP Net Income was $28.6 million or $0.34 per diluted share.",
        "key_terms": ["28.6", "0.34"]
    },
    {
        "id": "q16",
        "question": "What percentage of total revenue came from North America in 2024?",
        "answerable": True,
        "expected_pages": [4],
        "ground_truth": "North America accounted for $265.7 million or 62.0% of total revenue.",
        "key_terms": ["265.7", "62.0%"]
    },
    {
        "id": "q17",
        "question": "Which geographic region was the fastest growing for Novasphere in 2024 and at what rate?",
        "answerable": True,
        "expected_pages": [4],
        "ground_truth": "APAC was the fastest-growing region with an annual growth rate of 51.4%.",
        "key_terms": ["APAC", "51.4%"]
    },
    {
        "id": "q18",
        "question": "What was the largest industry vertical by subscription revenue and what percentage did it represent?",
        "answerable": True,
        "expected_pages": [4],
        "ground_truth": "Financial Services was the largest vertical, representing 38% of total subscription revenue.",
        "key_terms": ["Financial Services", "38%"]
    },
    {
        "id": "q19",
        "question": "In how many global cloud regions does Novasphere operate its SaaS infrastructure?",
        "answerable": True,
        "expected_pages": [5],
        "ground_truth": "Novasphere operates in 26 global cloud regions across three primary hyperscalers.",
        "key_terms": ["26"]
    },
    {
        "id": "q20",
        "question": "What consensus protocol is used across Novasphere's infrastructure?",
        "answerable": True,
        "expected_pages": [5],
        "ground_truth": "The consensus protocol used is Raft-Nova v2 (a modified Raft consensus).",
        "key_terms": ["Raft-Nova v2", "Raft"]
    },
    {
        "id": "q21",
        "question": "What was the total unscheduled downtime and system availability in 2024?",
        "answerable": True,
        "expected_pages": [5],
        "ground_truth": "System availability was 99.994% with total unscheduled downtime of 31.5 minutes.",
        "key_terms": ["99.994%", "31.5 minutes"]
    },
    {
        "id": "q22",
        "question": "What storage engine is used for local fast state in Novasphere's architecture?",
        "answerable": True,
        "expected_pages": [5],
        "ground_truth": "The RocksDB storage engine is used for local fast state.",
        "key_terms": ["RocksDB"]
    },
    {
        "id": "q23",
        "question": "What compliance certifications does Novasphere hold?",
        "answerable": True,
        "expected_pages": [6],
        "ground_truth": "Novasphere holds SOC 2 Type II, ISO/IEC 27001:2022, ISO/IEC 27701, HIPAA compliance, and FedRAMP Moderate In-Process.",
        "key_terms": ["SOC 2 Type II", "ISO/IEC 27001", "HIPAA", "FedRAMP"]
    },
    {
        "id": "q24",
        "question": "How often are data-at-rest encryption keys rotated at Novasphere?",
        "answerable": True,
        "expected_pages": [6],
        "ground_truth": "Data-at-rest encryption keys are rotated automatically every 90 days.",
        "key_terms": ["90 days", "AES-256"]
    },
    {
        "id": "q25",
        "question": "Which firm conducted Novasphere's external HIPAA compliance audit?",
        "answerable": True,
        "expected_pages": [6],
        "ground_truth": "Schellman & Company conducted the external HIPAA audit.",
        "key_terms": ["Schellman & Company"]
    },
    {
        "id": "q26",
        "question": "How many granted patents and pending patent applications did Novasphere hold as of 2024?",
        "answerable": True,
        "expected_pages": [7],
        "ground_truth": "Novasphere held 34 granted patents and 52 pending applications.",
        "key_terms": ["34", "52"]
    },
    {
        "id": "q27",
        "question": "What is the subject matter of US Patent 11,842,109?",
        "answerable": True,
        "expected_pages": [7],
        "ground_truth": "US Patent 11,842,109 covers 'Autonomous predictive scaling in heterogeneous cloud container environments'.",
        "key_terms": ["Autonomous predictive scaling", "heterogeneous cloud"]
    },
    {
        "id": "q28",
        "question": "Who is Novasphere's Chief Financial Officer and what was his previous role?",
        "answerable": True,
        "expected_pages": [8],
        "ground_truth": "The CFO is Marcus Vance, formerly the CFO at Datastream Corp.",
        "key_terms": ["Marcus Vance", "Datastream Corp"]
    },
    {
        "id": "q29",
        "question": "Who serves as the Chief Revenue Officer and when did she join?",
        "answerable": True,
        "expected_pages": [8],
        "ground_truth": "Sarah Chen-Boucher joined as Chief Revenue Officer in February 2023 from HashiCorp.",
        "key_terms": ["Sarah Chen-Boucher", "February 2023", "HashiCorp"]
    },
    {
        "id": "q30",
        "question": "Who is the independent Chairman of the Board of Directors?",
        "answerable": True,
        "expected_pages": [8],
        "ground_truth": "Arthur Pendelton serves as independent Chairman of the Board.",
        "key_terms": ["Arthur Pendelton"]
    },
    {
        "id": "q31",
        "question": "What is Novasphere's target year to achieve net-zero greenhouse gas emissions?",
        "answerable": True,
        "expected_pages": [9],
        "ground_truth": "Novasphere committed to achieving net-zero emissions by 2030.",
        "key_terms": ["2030", "net-zero"]
    },
    {
        "id": "q32",
        "question": "What certification did Novasphere's corporate offices in Seattle and Zurich achieve?",
        "answerable": True,
        "expected_pages": [9],
        "ground_truth": "The facilities achieved LEED Gold certification.",
        "key_terms": ["LEED Gold"]
    },
    {
        "id": "q33",
        "question": "What was the total global headcount of Novasphere as of 2024?",
        "answerable": True,
        "expected_pages": [9],
        "ground_truth": "The total global workforce was 876 employees across 14 countries.",
        "key_terms": ["876", "14 countries"]
    },
    {
        "id": "q34",
        "question": "What percentage of employees in technical engineering roles are women?",
        "answerable": True,
        "expected_pages": [9],
        "ground_truth": "Women represented 31% of technical engineering roles.",
        "key_terms": ["31%"]
    },
    {
        "id": "q35",
        "question": "What was the status of material pending litigation for Novasphere as of December 31, 2024?",
        "answerable": True,
        "expected_pages": [10],
        "ground_truth": "Novasphere was not party to any material pending litigation or regulatory enforcement actions.",
        "key_terms": ["not party", "no material pending litigation"]
    },

    # ── Unanswerable (15 items) ──────────────────────────────────────────────
    {
        "id": "q36",
        "question": "What was Novasphere's stock ticker price on Nasdaq in July 2024?",
        "answerable": False,
        "expected_pages": [],
        "ground_truth": "I couldn't find this in the document(s).",
        "key_terms": []
    },
    {
        "id": "q37",
        "question": "How much dividend per share did Novasphere pay out in the third quarter of 2024?",
        "answerable": False,
        "expected_pages": [],
        "ground_truth": "I couldn't find this in the document(s).",
        "key_terms": []
    },
    {
        "id": "q38",
        "question": "What is the name of Apple Inc.'s latest smartphone release?",
        "answerable": False,
        "expected_pages": [],
        "ground_truth": "I couldn't find this in the document(s).",
        "key_terms": []
    },
    {
        "id": "q39",
        "question": "Who is the Chief Marketing Officer at Novasphere Technologies?",
        "answerable": False,
        "expected_pages": [],
        "ground_truth": "I couldn't find this in the document(s).",
        "key_terms": []
    },
    {
        "id": "q40",
        "question": "What was Microsoft's Azure cloud revenue growth rate for 2024?",
        "answerable": False,
        "expected_pages": [],
        "ground_truth": "I couldn't find this in the document(s).",
        "key_terms": []
    },
    {
        "id": "q41",
        "question": "What are the details of Novasphere's acquisition of CloudBridge Networks in 2022?",
        "answerable": False,
        "expected_pages": [],
        "ground_truth": "I couldn't find this in the document(s).",
        "key_terms": []
    },
    {
        "id": "q42",
        "question": "How many total shares of common stock did CEO Aris Thorne sell during 2024?",
        "answerable": False,
        "expected_pages": [],
        "ground_truth": "I couldn't find this in the document(s).",
        "key_terms": []
    },
    {
        "id": "q43",
        "question": "What is the capital of Iceland?",
        "answerable": False,
        "expected_pages": [],
        "ground_truth": "I couldn't find this in the document(s).",
        "key_terms": []
    },
    {
        "id": "q44",
        "question": "Does Novasphere offer an on-premise hardware appliance called NovaRack 5000?",
        "answerable": False,
        "expected_pages": [],
        "ground_truth": "I couldn't find this in the document(s).",
        "key_terms": []
    },
    {
        "id": "q45",
        "question": "What was Novasphere's marketing spend specifically on Google Search Ads in 2024?",
        "answerable": False,
        "expected_pages": [],
        "ground_truth": "I couldn't find this in the document(s).",
        "key_terms": []
    },
    {
        "id": "q46",
        "question": "What is the recipe for chocolate chip cookies?",
        "answerable": False,
        "expected_pages": [],
        "ground_truth": "I couldn't find this in the document(s).",
        "key_terms": []
    },
    {
        "id": "q47",
        "question": "What server hardware manufacturer supplies the CPUs for Novasphere's internal clusters?",
        "answerable": False,
        "expected_pages": [],
        "ground_truth": "I couldn't find this in the document(s).",
        "key_terms": []
    },
    {
        "id": "q48",
        "question": "What was the inflation rate in the Eurozone in December 2024?",
        "answerable": False,
        "expected_pages": [],
        "ground_truth": "I couldn't find this in the document(s).",
        "key_terms": []
    },
    {
        "id": "q49",
        "question": "Who won the FIFA World Cup in 2022?",
        "answerable": False,
        "expected_pages": [],
        "ground_truth": "I couldn't find this in the document(s).",
        "key_terms": []
    },
    {
        "id": "q50",
        "question": "What was the date of Novasphere's initial public offering on the New York Stock Exchange?",
        "answerable": False,
        "expected_pages": [],
        "ground_truth": "I couldn't find this in the document(s).",
        "key_terms": []
    }
]


def create_qa_jsonl() -> Path:
    """Write the 50 QA pairs to JSONL."""
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    with QA_JSONL_PATH.open("w", encoding="utf-8") as f:
        for item in QA_PAIRS:
            f.write(json.dumps(item) + "\n")
    return QA_JSONL_PATH


def main():
    pdf = create_sample_pdf()
    qa = create_qa_jsonl()
    print(f"Generated PDF: {pdf} (10 pages)")
    print(f"Generated QA JSONL: {qa} (50 pairs: 35 answerable, 15 unanswerable)")


if __name__ == "__main__":
    main()
