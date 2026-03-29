# Ground truth evaluation dataset derived from microsoft.pdf (10-K FY2024)
# Each entry has:
#   - question:         exact question asked
#   - reference_answer: manually verified correct answer from the document
#   - relevant_pages:   1-based page numbers that contain the answer (for retriever eval)

EVAL_DATASET = [
    {
        "question": "What was Microsoft's net income in 2024?",
        "reference_answer": "Microsoft's net income in fiscal year 2024 was $88,308 million.",
        "relevant_pages": [77, 47],
    },
    {
        "question": "What is the company's stated mission or vision according to the document?",
        "reference_answer": (
            "Microsoft's mission is to empower every person and every organization "
            "on the planet to achieve more."
        ),
        "relevant_pages": [39],
    },
    {
        "question": "What is the dividend per share declared in the most recent financial statement?",
        "reference_answer": (
            "The total cash dividend declared per common share for fiscal year 2024 was $3.00, "
            "paid in four quarterly instalments of $0.75 each."
        ),
        "relevant_pages": [60, 89],
    },
    {
        "question": "On what date does Microsoft's fiscal year end?",
        "reference_answer": "Microsoft's fiscal year ends on June 30.",
        "relevant_pages": [39, 95],
    },
    {
        "question": "How does Microsoft break down its research and development spending?",
        "reference_answer": (
            "Research and development expenses include payroll, employee benefits, "
            "stock-based compensation expense, other headcount-related expenses associated "
            "with product development, third-party development and programming costs, and "
            "the amortization of purchased software code and services."
        ),
        "relevant_pages": [65],
    },
    {
        "question": "How does Microsoft describe its cloud strategy?",
        "reference_answer": (
            "Microsoft's cloud strategy centres on building the Intelligent Cloud and Intelligent "
            "Edge Platform. It benefits from three economies of scale: lower-cost datacenters, "
            "coordinated and aggregated demand patterns that improve resource utilisation, and "
            "multi-tenancy that lowers maintenance labour costs. Microsoft also serves as "
            "OpenAI's exclusive cloud provider and offers Azure Arc for consistent multi-cloud "
            "and on-premises management."
        ),
        "relevant_pages": [5, 39],
    },
    {
        "question": "What are the geographic regions where Microsoft generates the most revenue?",
        "reference_answer": (
            "In fiscal year 2024 the United States generated $124,704 million in revenue and "
            "all other countries combined generated $120,418 million. No individual country "
            "outside the United States accounted for more than 10% of total revenue."
        ),
        "relevant_pages": [94],
    },
    {
        "question": "What are the main risks Microsoft identifies in its 10-K?",
        "reference_answer": (
            "Key risks include: intense competition and rapid industry change; regulatory and legal "
            "challenges including the EU Digital Markets Act; cybersecurity threats; claims and "
            "lawsuits; intellectual property risks; data privacy and compliance obligations; "
            "macroeconomic conditions; and risks related to AI development and deployment."
        ),
        "relevant_pages": [20, 21, 29, 31],
    },
    {
        "question": "What are the top three products or services by revenue, as listed in the report?",
        "reference_answer": (
            "For fiscal year 2024 the top three by revenue were: "
            "1. Server products and cloud services – $97,726 million. "
            "2. Office products and cloud services – $54,875 million. "
            "3. Windows – $23,244 million."
        ),
        "relevant_pages": [94],
    },
    {
        "question": "What legal proceedings or regulatory challenges are mentioned in the latest filings?",
        "reference_answer": (
            "Microsoft faces claims and lawsuits across product, employment, and AI practices; "
            "EU Digital Markets Act and Digital Services Act compliance requirements; "
            "data privacy and telecommunications regulation; intellectual property allegations; "
            "environmental and sustainability disclosure requirements; and a cell-phone brain "
            "cancer lawsuit where plaintiffs' expert testimony was excluded."
        ),
        "relevant_pages": [30, 31, 32, 87],
    },
    {
        "question": "What partnerships or collaborations are highlighted in the document?",
        "reference_answer": (
            "Highlighted partnerships include: the Microsoft Partner Network and Cloud Solution "
            "Provider Program for enterprise deployments; OpenAI as Microsoft's exclusive cloud "
            "partner; the TEALS programme (partnering with ~1,500 volunteers across ~550 high "
            "schools); and a $150 million commitment to Minority Depository Institutions and "
            "Black and African American-owned small businesses."
        ),
        "relevant_pages": [8, 11],
    },
    {
        "question": "What was the date of Microsoft's most recent major acquisition, and what company was acquired?",
        "reference_answer": (
            "Microsoft acquired Activision Blizzard. The acquisition closed during fiscal year 2024 "
            "(the pro forma financials treat it as consummated on July 1, 2022 for comparison purposes)."
        ),
        "relevant_pages": [77],
    },
]
