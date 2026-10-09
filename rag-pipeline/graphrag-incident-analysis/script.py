from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
import os, re

# Load environment variables from .env file
load_dotenv()


# ============================================================
# 1. CONFIGURATION
# ============================================================

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


CHROMA_DIR = "./chroma_db"
LLM_MODEL = "qwen3:8b"
EMBEDDING_MODEL = "nomic-embed-text"

# Neo4j data extraction custom insructions
CYPHER_GENERATION_TEMPLATE = """
You are an expert Neo4j Cypher query writer.

Generate a read-only Cypher query using only the provided schema.
Fix the syntax and structure of the query if it is invalid.

Schema:
{schema}

Question:
{question}

Cypher query:
"""


# ============================================================
# 2. INITIALIZE OLLAMA
# ============================================================

llm = OllamaLLM(
    model=LLM_MODEL,
    temperature=0,
)

embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL,
)


# ============================================================
# 3. CONNECT TO NEO4J
# ============================================================

graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
)

graph.refresh_schema()


# ============================================================
# 4. INITIALIZE GRAPHCYPHERQACHAIN
# ============================================================
cypher_prompt = PromptTemplate(
    input_variables=["schema", "question"],
    template=CYPHER_GENERATION_TEMPLATE,
)

graph_chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    cypher_prompt=cypher_prompt,
    verbose=True,
    allow_dangerous_requests=True,
    validate_cypher=True,
    return_intermediate_steps=True,
    top_k=20,
)


# ============================================================
# 5. INITIALIZE CHROMADB
# ============================================================

vectorstore = Chroma(
    collection_name="devops_graphrag",
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR,
)

# ============================================================
# 6. QUERY NEO4J USING GRAPHCYPHERQACHAIN
# ============================================================

def query_graph(question: str):

    result = graph_chain.invoke({
        "query": question,
    })

    print("\nGenerated Cypher and intermediate results:")

    intermediate_steps = result.get(
        "intermediate_steps",
        [],
    )

    for step in intermediate_steps:
        print(step)

    # Return the complete chain result so we can inspect
    # the graph answer and intermediate query results.
    return result


# ============================================================
# 7. EXTRACT DOCUMENT IDS FROM GRAPH RESULTS
# ============================================================

def extract_document_ids(graph_result):
    document_ids = re.findall(r'\b[a-z]+(?:-[a-z0-9]+)+\b', graph_result)
    
    return list(set(document_ids))


# ============================================================
# 8. GRAPH-GUIDED CHROMA RETRIEVAL
# ============================================================

def retrieve_documents(question: str, document_ids: list[str]):

    if not document_ids:
        print(
            "\nNo document IDs were returned by Neo4j. "
            "Skipping ChromaDB retrieval."
        )
        return []

    print("\nDocument IDs selected by Neo4j:")

    for document_id in document_ids:
        print(f"  - {document_id}")

    # Apply metadata filtering BEFORE vector similarity search.
    retrieved_docs = vectorstore.similarity_search(
        query=question,
        k=5,
        filter={
            "documentId": {
                "$in": document_ids
            }
        },
    )

    return retrieved_docs


# ============================================================
# 9. FINAL ANSWER PROMPT
# ============================================================

final_prompt = ChatPromptTemplate.from_template("""
You are a DevOps incident investigation assistant.

Answer the user's question using the retrieved graph and
documentation evidence.

GRAPH ANSWER:
{graph_answer}

RETRIEVED DOCUMENTATION:
{document_context}

USER QUESTION:
{question}

Instructions:
- Use the documentation as supporting evidence.
- Do not invent facts or root causes.
- Distinguish confirmed causes from possible causes.
- If evidence is insufficient, say so.

Provide:
1. Explanation
2. Supporting evidence
3. Recommended investigation steps
""")


# ============================================================
# 10. COMPLETE GRAPHRAG PIPELINE
# ============================================================

def ask(question: str):

    print("\n================================")
    print("USER QUESTION")
    print("================================")

    print(question)

    # --------------------------------------------------------
    # STEP 1: QUERY NEO4J
    # --------------------------------------------------------

    print("\n================================")
    print("STEP 1: GRAPHCYPHERQACHAIN")
    print("================================")

    graph_result = query_graph(question)

    graph_answer = graph_result.get(
        "result",
        "",
    )

    # --------------------------------------------------------
    # STEP 2: EXTRACT DOCUMENT IDS
    # --------------------------------------------------------

    print("\n================================")
    print("STEP 2: EXTRACT DOCUMENT IDS")
    print("================================")

    document_ids = extract_document_ids(
        graph_result["result"]
    )

    print("\nExtracted document IDs from graph results:")
    print(document_ids)
    print("*" *90)

    # --------------------------------------------------------
    # STEP 3: SEARCH CHROMADB USING METADATA FILTER
    # --------------------------------------------------------

    print("\n================================")
    print("STEP 3: CHROMADB RETRIEVAL")
    print("================================")

    retrieved_docs = retrieve_documents(
        question,
        document_ids,
    )

    for doc in retrieved_docs:
        print(
            f"{doc.metadata.get('documentId')} "
            f"| Page {doc.metadata.get('page')}"
        )

    # --------------------------------------------------------
    # STEP 4: BUILD DOCUMENT CONTEXT
    # --------------------------------------------------------

    document_context = "\n\n".join(
        f"""
Document ID: {doc.metadata.get('documentId')}
Source: {doc.metadata.get('sourceFile')}
Page: {doc.metadata.get('page')}

{doc.page_content}
"""
        for doc in retrieved_docs
    )

    if not document_context:
        document_context = (
            "No supporting PDF chunks were retrieved."
        )

    # --------------------------------------------------------
    # STEP 5: FINAL LLM INFERENCE
    # --------------------------------------------------------

    print("\n================================")
    print("STEP 4: QWEN FINAL INFERENCE")
    print("================================")

    messages = final_prompt.invoke({
        "graph_answer": graph_answer,
        "document_context": document_context,
        "question": question,
    })

    answer = llm.invoke(messages)

    print("\n================================")
    print("FINAL ANSWER")
    print("================================")

    print(answer)

    return answer


# ============================================================
# 13. MAIN
# ============================================================

if __name__ == "__main__":

    ask(
        "Why is PaymentService restarting after deployment, and which documents should I investigate?"
    )



# Output
'''

================================
USER QUESTION
================================
Why is PaymentService restarting after deployment, and which documents should I investigate?

================================
STEP 1: GRAPHCYPHERQACHAIN
================================


> Entering new GraphCypherQAChain chain...
Generated Cypher:
// Find incidents affecting PaymentService, their root causes, related databases, and deployment documents
MATCH (paymentService:Service {name: 'PaymentService'})
// Find incidents affecting the service
MATCH (incident:Incident)-[:AFFECTS]->(paymentService)
// Get root causes of each incident
MATCH (incident)-[:CAUSED_BY]->(rootCause:RootCause)
// Find databases related to root causes
MATCH (rootCause)-[:RELATED_TO]->(database:Database)
// Get deployment details for PaymentService
MATCH (paymentService)-[:HAS_DEPLOYMENT]->(deployment:Deployment)
// Collect all relevant document IDs
RETURN 
  incident.name AS incidentName,
  rootCause.name AS rootCauseName,
  database.name AS databaseName,
  deployment.documentIds AS deploymentDocuments,
  paymentService.documentIds AS serviceDocuments

Full Context:
[{'incidentName': 'Payment pods repeatedly restarting', 'rootCauseName': 'PostgreSQL transaction timeout', 'databaseName': 'Payment PostgreSQL', 'deploymentDocuments': ['kubernetes-payment-deployment', 'payment-service-runbook'], 'serviceDocuments': ['payment-service-runbook', 'payment-service-architecture', 'payment-incident-0814', 'payment-incident-0821']}, {'incidentName': 'Payment transaction timeout', 'rootCauseName': 'PostgreSQL connection pool exhaustion', 'databaseName': 'Payment PostgreSQL', 'deploymentDocuments': ['kubernetes-payment-deployment', 'payment-service-runbook'], 'serviceDocuments': ['payment-service-runbook', 'payment-service-architecture', 'payment-incident-0814', 'payment-incident-0821']}]

> Finished chain.

Generated Cypher and intermediate results:
{'query': "// Find incidents affecting PaymentService, their root causes, related databases, and deployment documents\nMATCH (paymentService:Service {name: 'PaymentService'})\n// Find incidents affecting the service\nMATCH (incident:Incident)-[:AFFECTS]->(paymentService)\n// Get root causes of each incident\nMATCH (incident)-[:CAUSED_BY]->(rootCause:RootCause)\n// Find databases related to root causes\nMATCH (rootCause)-[:RELATED_TO]->(database:Database)\n// Get deployment details for PaymentService\nMATCH (paymentService)-[:HAS_DEPLOYMENT]->(deployment:Deployment)\n// Collect all relevant document IDs\nRETURN \n  incident.name AS incidentName,\n  rootCause.name AS rootCauseName,\n  database.name AS databaseName,\n  deployment.documentIds AS deploymentDocuments,\n  paymentService.documentIds AS serviceDocuments\n"}
{'context': [{'incidentName': 'Payment pods repeatedly restarting', 'rootCauseName': 'PostgreSQL transaction timeout', 'databaseName': 'Payment PostgreSQL', 'deploymentDocuments': ['kubernetes-payment-deployment', 'payment-service-runbook'], 'serviceDocuments': ['payment-service-runbook', 'payment-service-architecture', 'payment-incident-0814', 'payment-incident-0821']}, {'incidentName': 'Payment transaction timeout', 'rootCauseName': 'PostgreSQL connection pool exhaustion', 'databaseName': 'Payment PostgreSQL', 'deploymentDocuments': ['kubernetes-payment-deployment', 'payment-service-runbook'], 'serviceDocuments': ['payment-service-runbook', 'payment-service-architecture', 'payment-incident-0814', 'payment-incident-0821']}]}

================================
STEP 2: EXTRACT DOCUMENT IDS
================================

Extracted document IDs from graph results:
['payment-incident-0821', 'payment-service-runbook', 'payment-service-architecture', 'kubernetes-payment-deployment', 'payment-incident-0814']
******************************************************************************************

================================
STEP 3: CHROMADB RETRIEVAL
================================

Document IDs selected by Neo4j:
  - payment-incident-0821
  - payment-service-runbook
  - payment-service-architecture
  - kubernetes-payment-deployment
  - payment-incident-0814
payment-service-runbook | Page 0
payment-service-runbook | Page 0
kubernetes-payment-deployment | Page 0
kubernetes-payment-deployment | Page 0
payment-incident-0814 | Page 0

================================
STEP 4: QWEN FINAL INFERENCE
================================

================================
FINAL ANSWER
================================
### 1. Explanation  
PaymentService is restarting after deployment due to **PostgreSQL transaction timeouts** and **connection pool exhaustion**, which cause the service to become unhealthy. These issues likely trigger restarts when the service cannot handle database operations or connections efficiently. The root cause is tied to database connectivity and resource utilization, as confirmed by historical incidents and deployment documentation.  

---

### 2. Supporting Evidence  
- **Graph Answer**:  
  - Confirmed root causes: PostgreSQL transaction timeout and connection pool exhaustion.  
  - Affected service: PaymentService.  
  - Related documents: `payment-service-runbook`, `kubernetes-payment-deployment`, `payment-incident-0814`.  

- **Documentation Evidence**:  
  - **`payment-service-runbook`**:  
    - Indicates that **database transaction timeouts** (e.g., PostgreSQL) directly correlate with PaymentService pod restarts (Q: "What should I check when PaymentService pods keep restarting?").  
    - Lists dependencies including **Payment PostgreSQL**, which must be validated for connectivity and resource usage.  
  - **`kubernetes-payment-deployment`**:  
    - Confirms the current deployment version (`v2.4.1`) and replica count (3 pods).  
    - Highlights checks for **database connectivity** and **health probes** after deployment changes.  
  - **`payment-incident-0814`**:  
    - Directly links **PostgreSQL transaction timeouts** to PaymentService pod restarts in a past incident (INC-2026-0814).  
    - Shows that liveness probe failures and pod restarts were the primary symptoms.  

---

### 3. Recommended Investigation Steps  
1. **Check Deployment Rollout and Pod Status**:  
   - Verify the rollout status of `payment-service-v2.4.1` (from `kubernetes-payment-deployment`).  
   - Confirm if all three replicas are healthy or if one is unhealthy (as seen in INC-2026-0814).  

2. **Analyze Container Logs**:  
   - Look for **PostgreSQL transaction timeout errors** or **connection pool exhaustion** in PaymentService logs (from `payment-service-runbook`).  
   - Validate **liveness probe failures** and **database connection errors** (e.g., timeouts, refused connections).  

3. **Investigate PostgreSQL Health**:  
   - Check PostgreSQL **resource utilization** (CPU, memory, disk I/O) and **active transaction counts**.  
   - Ensure **connection pool settings** (e.g., max connections) are aligned with workload demands.  

4. **Review Recent Deployment Changes**:  
   - Cross-reference deployment changes (e.g., config updates, version upgrades) with the incident timeline (from `payment-incident-0814`).  

5. **Validate Health Probes**:  
   - Confirm that **readiness and liveness probes** are configured correctly and not triggering false restarts.  

6. **Compare with Historical Incident**:  
   - Review the resolution steps from **INC-2026-0814** (e.g., PostgreSQL timeout mitigation) to identify patterns or recurring issues.  

---  
**Documents to Investigate**:  
- `payment-service-runbook` (troubleshooting steps and dependencies).  
- `kubernetes-payment-deployment` (deployment configuration and version details).  
- `payment-incident-0814` (historical root cause and resolution for PostgreSQL-related restarts).  

If evidence is insufficient, further analysis of PostgreSQL logs and metrics may be required.
'''