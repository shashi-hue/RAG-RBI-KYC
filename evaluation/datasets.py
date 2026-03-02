"""
evaluation/datasets.py
All manually-authored eval records in one place.

TYPES:
  faq       — from FAQ PDF (37 Q&A, expected_chunk_ids filled after step 3)
  deep      — deep regulatory questions from the Master Direction body text
  multihop  — questions requiring 2+ chunk combination
  intent    — router classification tests
  negative  — out-of-scope / refusal tests

expected_chunk_ids: fill these AFTER running generate_candidates.py (step 3).
"""

# ── DEEP QA (from Master Direction body — NOT in FAQ) ─────────────────────────
DEEP_QA = [
    {
        "id": "deep_001", "type": "deep",
        "question": "What is the beneficial ownership threshold for a company under the KYC Master Direction?",
        "reference_answer": "Controlling ownership interest means ownership of or entitlement to more than 10 percent of the shares or capital or profits of the company.",
        "expected_chunk_ids": [],
        "hint": "Para 3, Chapter I",
    },
    {
        "id": "deep_002", "type": "deep",
        "question": "What is the beneficial ownership threshold for a partnership firm?",
        "reference_answer": "More than 10 percent of capital or profits of the partnership.",
        "expected_chunk_ids": [],
        "hint": "Para 3, Chapter I",
    },
    {
        "id": "deep_003", "type": "deep",
        "question": "What is the beneficial ownership threshold for an unincorporated association or body of individuals?",
        "reference_answer": "More than 15 percent of the property or capital or profits.",
        "expected_chunk_ids": [],
        "hint": "Para 3, Chapter I",
    },
    {
        "id": "deep_004", "type": "deep",
        "question": "Who qualifies as beneficial owner when the customer is a trust?",
        "reference_answer": "The author of the trust, the trustee, beneficiaries with 10 percent or more interest, and any person exercising ultimate effective control.",
        "expected_chunk_ids": [],
        "hint": "Para 3, Chapter I",
    },
    {
        "id": "deep_005", "type": "deep",
        "question": "What is a shell bank under the KYC directions?",
        "reference_answer": "A bank with no physical presence in the country where it is incorporated and licensed, and unaffiliated with a regulated financial group subject to effective consolidated supervision.",
        "expected_chunk_ids": [],
        "hint": "Para 3, Chapter I",
    },
    {
        "id": "deep_006", "type": "deep",
        "question": "What does ongoing due diligence mean under the KYC directions?",
        "reference_answer": "Regular monitoring of transactions to ensure those are consistent with the RE's knowledge about the customer's business, risk profile, and source of funds or wealth.",
        "expected_chunk_ids": [],
        "hint": "Para 3, Chapter I",
    },
    {
        "id": "deep_007", "type": "deep",
        "question": "What is a suspicious transaction as defined in the Master Direction?",
        "reference_answer": "A transaction giving rise to reasonable ground of suspicion of ML/TF proceeds, unusual complexity, no economic rationale, or financing of terrorism.",
        "expected_chunk_ids": [],
        "hint": "Para 3, Chapter I",
    },
    {
        "id": "deep_008", "type": "deep",
        "question": "How are branches of Indian banks abroad required to comply with KYC standards?",
        "reference_answer": "They must adopt the more stringent of RBI and host country standards. If host country laws prohibit RBI guidelines, the matter must be brought to RBI's notice.",
        "expected_chunk_ids": [],
        "hint": "Para 2, Chapter I",
    },
    {
        "id": "deep_009", "type": "deep",
        "question": "What is a payable-through account under the KYC directions?",
        "reference_answer": "Correspondent accounts used directly by third parties to transact business on their own behalf.",
        "expected_chunk_ids": [],
        "hint": "Para 3, Chapter I",
    },
    {
        "id": "deep_010", "type": "deep",
        "question": "What is Digital KYC as defined in the Master Direction?",
        "reference_answer": "Capturing a live photo of the customer and the OVD along with the latitude and longitude of the location where the photo is taken by an authorised officer of the RE.",
        "expected_chunk_ids": [],
        "hint": "Para 3, Chapter I",
    },
    {
        "id": "deep_011", "type": "deep",
        "question": "What is an equivalent e-document under the KYC directions?",
        "reference_answer": "An electronic equivalent of a document issued by the issuing authority with its valid digital signature, including documents issued via DigiLocker.",
        "expected_chunk_ids": [],
        "hint": "Para 3, Chapter I",
    },
    {
        "id": "deep_012", "type": "deep",
        "question": "What is a cross-border wire transfer under the Master Direction?",
        "reference_answer": "Any wire transfer where the ordering and beneficiary financial institutions are in different countries, including any chain where at least one institution is abroad.",
        "expected_chunk_ids": [],
        "hint": "Para 3, Chapter I wire transfer definitions",
    },
    {
        "id": "deep_013", "type": "deep",
        "question": "What is the definition of Customer Due Diligence (CDD) under the KYC directions?",
        "reference_answer": "Identifying and verifying the customer and the beneficial owner using reliable and independent sources, including understanding the nature of business and determining if the customer acts on behalf of a beneficial owner.",
        "expected_chunk_ids": [],
        "hint": "Para 3, Chapter I",
    },
    {
        "id": "deep_014", "type": "deep",
        "question": "What is CKYCR and what is its function?",
        "reference_answer": "Central KYC Records Registry — an entity to receive, store, safeguard and retrieve the KYC records in digital form of a customer.",
        "expected_chunk_ids": [],
        "hint": "Para 3, Chapter I",
    },
    {
        "id": "deep_015", "type": "deep",
        "question": "What is Periodic Updation as defined in the KYC directions?",
        "reference_answer": "Steps taken to ensure documents, data or information collected under the CDD process is kept up-to-date and relevant by undertaking reviews at the periodicity prescribed by the Reserve Bank.",
        "expected_chunk_ids": [],
        "hint": "Para 3, Chapter I",
    },
]

# ── MULTI-HOP QA (requires combining 2+ chunks) ───────────────────────────────
MULTIHOP_QA = [
    {
        "id": "mh_001", "type": "multihop",
        "question": "What is the re-KYC frequency difference between high-risk and low-risk customers, and what documents does each require?",
        "reference_answer": "High-risk: at least every 2 years. Low-risk: every 10 years. If info changed, OVDs required; if no change, self-declaration suffices.",
        "expected_chunk_ids": [],
        "hint": "FAQ Q22 + FAQ Q25",
    },
    {
        "id": "mh_002", "type": "multihop",
        "question": "Can a person with disability use V-CIP to open an account, and are specific facial gestures required during liveness check?",
        "reference_answer": "Yes, V-CIP is available to PwDs. Specific facial gestures like blinking or smiling are NOT mandatory; the RE must consider the customer's special needs.",
        "expected_chunk_ids": [],
        "hint": "FAQ Q33 + FAQ Q20",
    },
    {
        "id": "mh_003", "type": "multihop",
        "question": "When a customer uses their KYC Identifier to open an account, in what circumstances must they still submit fresh KYC documents?",
        "reference_answer": "When info in CKYCR has changed, record is incomplete or not per current norms, validity of documents has lapsed, or RE needs to verify identity/address or perform EDD.",
        "expected_chunk_ids": [],
        "hint": "FAQ Q18",
    },
    {
        "id": "mh_004", "type": "multihop",
        "question": "What are the modes for reactivating an inoperative account and what defines an inoperative account?",
        "reference_answer": "Inoperative account: savings/current with no customer-induced transactions for over 2 years. Reactivation: KYC updation at any branch or via V-CIP (if available).",
        "expected_chunk_ids": [],
        "hint": "FAQ Q36 + FAQ Q37",
    },
    {
        "id": "mh_005", "type": "multihop",
        "question": "Can a customer whose OVD shows a Delhi address open an account in Chennai, and what must they do within 3 months?",
        "reference_answer": "Yes, by submitting a deemed OVD for proof of address. Within 3 months they must submit an OVD with current address.",
        "expected_chunk_ids": [],
        "hint": "FAQ Q7 + FAQ Q8",
    },
    {
        "id": "mh_006", "type": "multihop",
        "question": "Is Aadhaar mandatory for KYC, and can it be used for non-face-to-face periodic KYC updation?",
        "reference_answer": "Aadhaar is not mandatory unless receiving government subsidies under Section 7. For NF2F periodic updation, Aadhaar OTP-based e-KYC is one of the permitted modes.",
        "expected_chunk_ids": [],
        "hint": "FAQ Q10 + FAQ Q24",
    },
    {
        "id": "mh_007", "type": "multihop",
        "question": "What is the beneficial ownership threshold for a company versus a trust?",
        "reference_answer": "Company: more than 10% shareholding/capital/profits. Trust: author, trustee, beneficiaries with 10% or more interest, and any person with ultimate effective control.",
        "expected_chunk_ids": [],
        "hint": "Para 3 Chapter I — company BO + trust BO",
    },
    {
        "id": "mh_008", "type": "multihop",
        "question": "What warning has RBI issued about KYC fraud and what law allows account closure for non-compliance?",
        "reference_answer": "RBI warned against clicking suspicious links in KYC SMS/emails (Press Release Sep 13, 2021). PMLA Rules 2005 enable account closure after due notice for non-compliance.",
        "expected_chunk_ids": [],
        "hint": "FAQ Q29 + FAQ Q28",
    },
    {
        "id": "mh_009", "type": "multihop",
        "question": "Can a KYC or periodic KYC updation application be automatically rejected, and what is the rule for joint accounts?",
        "reference_answer": "Automated rejection is not permitted; decisions must be reviewed by an authorised RE official. All joint account holders must submit KYC documents individually.",
        "expected_chunk_ids": [],
        "hint": "FAQ Q34 + FAQ Q12",
    },
    {
        "id": "mh_010", "type": "multihop",
        "question": "Can an NRI submit KYC documents without visiting a branch, and who can certify their OVD copies?",
        "reference_answer": "Yes, NRIs can use non-face-to-face onboarding. OVD copies can be certified by overseas branches of Indian SCBs, notary public, court magistrate, judge, or Indian embassy/consulate.",
        "expected_chunk_ids": [],
        "hint": "FAQ Q13 + Certified Copy NRI definition Chapter I",
    },
]

# ── INTENT QA ─────────────────────────────────────────────────────────────────
# expected_intent values must match QueryIntent enum: fpidocs, chapter, historical, general
INTENT_QA = [
    {"id": "int_001", "type": "intent", "question": "What is KYC?",                                              "expected_intent": "general"},
    {"id": "int_002", "type": "intent", "question": "Is KYC mandatory?",                                         "expected_intent": "general"},
    {"id": "int_003", "type": "intent", "question": "What is a suspicious transaction?",                          "expected_intent": "general"},
    {"id": "int_004", "type": "intent", "question": "What is a small account?",                                   "expected_intent": "general"},
    {"id": "int_005", "type": "intent", "question": "What documents are needed to open a bank account?",          "expected_intent": "general"},
    {"id": "int_006", "type": "intent", "question": "How do I update my KYC if my address has changed?",          "expected_intent": "general"},
    {"id": "int_007", "type": "intent", "question": "What documents does a Category III FPI need to submit?",     "expected_intent": "fpidocs"},
    {"id": "int_008", "type": "intent", "question": "List KYC requirements for a foreign portfolio investor.",    "expected_intent": "fpidocs"},
    {"id": "int_009", "type": "intent", "question": "What KYC documents are required for FPI entity-level KYC?", "expected_intent": "fpidocs"},
    {"id": "int_010", "type": "intent", "question": "What was the old provision for customer identification before it was deleted?", "expected_intent": "historical"},
    {"id": "int_011", "type": "intent", "question": "What are the deleted provisions related to wire transfers?", "expected_intent": "historical"},
    {"id": "int_012", "type": "intent", "question": "What did the former text of paragraph 38 say?",              "expected_intent": "historical"},
    {"id": "int_013", "type": "intent", "question": "What are the rules about wire transfers and correspondent banking?", "expected_intent": "chapter"},
    {"id": "int_014", "type": "intent", "question": "What does Chapter VI say about enhanced due diligence?",     "expected_intent": "chapter"},
    {"id": "int_015", "type": "intent", "question": "What are the record maintenance requirements under Chapter VII?", "expected_intent": "chapter"},
]

# ── NEGATIVE / OUT-OF-SCOPE QA ────────────────────────────────────────────────
NEGATIVE_QA = [
    {"id": "neg_001", "type": "negative", "question": "What is the current repo rate set by RBI?"},
    {"id": "neg_002", "type": "negative", "question": "What is the GST rate on banking services?"},
    {"id": "neg_003", "type": "negative", "question": "How do I apply for a home loan?"},
    {"id": "neg_004", "type": "negative", "question": "What is India's GDP growth rate in 2025?"},
    {"id": "neg_005", "type": "negative", "question": "How do I invest in mutual funds?"},
    {"id": "neg_006", "type": "negative", "question": "What is the penalty for income tax evasion?"},
    {"id": "neg_007", "type": "negative", "question": "Tell me a joke about banking."},
    {"id": "neg_008", "type": "negative", "question": "What is the interest rate on SBI savings accounts?"},
    {"id": "neg_009", "type": "negative", "question": "How do I reset my net banking password?"},
    {"id": "neg_010", "type": "negative", "question": "What are SEBI regulations for mutual funds?"},
]