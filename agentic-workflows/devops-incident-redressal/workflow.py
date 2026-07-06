from typing import Literal
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama

from state import RouterState
from tools import (
    kubectl_get_pods,
    kubectl_describe_pod,
    kubectl_logs,
    kubectl_get_events,
    db_connection_stats,
    db_slow_queries,
    network_ping,
    dns_lookup,
    ci_pipeline_status,
    rollback_deployment
)

# =========================================================
# LLM (Local Qwen via Ollama)
# =========================================================

llm = ChatOllama(
    model="qwen3:8b",
    temperature=0
)

# =========================================================
# 1. ROUTER NODE (LLM decides which specialist)
# =========================================================

def router_node(state: RouterState):
    """
    LLM classifies incident into:
    kubernetes | database | network | cicd
    """

    prompt = f"""
You are a DevOps incident routing system.

Classify the incident into exactly ONE category:

- kubernetes → pod crashes, restarts, deployments, services
- database → slow queries, connections, latency, postgres issues
- network → DNS, latency, connectivity, external access
- cicd → pipeline failures, deployment issues, rollback needs

Return ONLY one word.

Incident:
{state["incident"]}
"""

    response = llm.invoke(prompt).content.strip().lower()

    # Normalize LLM output safely
    if "kubernetes" in response:
        route = "kubernetes"
    elif "database" in response:
        route = "database"
    elif "network" in response:
        route = "network"
    else:
        route = "cicd"

    print(f"\n🧭 ROUTER → {route}\n")

    return {
        **state,
        "route": route
    }


# =========================================================
# 2. KUBERNETES AGENT
# =========================================================

def kubernetes_agent(state: RouterState):
    incident = state["incident"]

    print("\n🚢 Kubernetes Agent running...\n")

    # Extract service name (simple heuristic)
    service = "payment-service"
    if "auth" in incident:
        service = "auth-service"
    elif "order" in incident:
        service = "order-service"

    pods = kubectl_get_pods.invoke(service)
    logs = kubectl_logs.invoke(service)
    events = kubectl_get_events.invoke("cluster")

    diagnosis = f"""
Kubernetes Issue Detected for {service}

Findings:
{pods}
"""

    commands = [
        f"kubectl get pods -n production | grep {service}",
        f"kubectl logs deployment/{service}",
        f"kubectl describe pod {service}-pod-1",
        "kubectl get events --sort-by=.lastTimestamp"
    ]

    return {
        **state,
        "diagnosis": diagnosis,
        "commands": commands,
        "evidence": [logs, events]
    }


# =========================================================
# 3. DATABASE AGENT
# =========================================================

def database_agent(state: RouterState):
    print("\n🗄️ Database Agent running...\n")

    stats = db_connection_stats.invoke("")
    slow = db_slow_queries.invoke("")

    diagnosis = f"""
Database Performance Issue Detected

{stats}
{slow}
"""

    commands = [
        "SELECT * FROM pg_stat_activity;",
        "SELECT * FROM pg_stat_statements ORDER BY total_exec_time DESC;",
        "VACUUM ANALYZE;"
    ]

    return {
        **state,
        "diagnosis": diagnosis,
        "commands": commands,
        "evidence": [stats, slow]
    }


# =========================================================
# 4. NETWORK AGENT
# =========================================================

def network_agent(state: RouterState):
    print("\n🌐 Network Agent running...\n")

    incident = state["incident"]

    domain = "api.company.com"
    if "auth" in incident:
        domain = "auth.company.com"

    ping = network_ping.invoke(domain)
    dns = dns_lookup.invoke(domain)

    diagnosis = f"""
Network Issue Detected

{ping}
{dns}
"""

    commands = [
        f"ping {domain}",
        f"nslookup {domain}",
        f"curl -I https://{domain}",
        "traceroute " + domain
    ]

    return {
        **state,
        "diagnosis": diagnosis,
        "commands": commands,
        "evidence": [ping, dns]
    }


# =========================================================
# 5. CI/CD AGENT
# =========================================================

def cicd_agent(state: RouterState):
    print("\n🚀 CI/CD Agent running...\n")

    incident = state["incident"]

    pipeline = "payment-deploy"
    if "auth" in incident:
        pipeline = "auth-deploy"

    status = ci_pipeline_status.invoke(pipeline)

    diagnosis = f"""
CI/CD Issue Detected

{status}
"""

    commands = [
        f"gh run list --workflow={pipeline}",
        "kubectl rollout status deployment/payment-service",
        "kubectl rollout undo deployment/payment-service"
    ]

    return {
        **state,
        "diagnosis": diagnosis,
        "commands": commands,
        "evidence": [status]
    }


# =========================================================
# 6. FINAL INCIDENT REPORT NODE
# =========================================================

def response_node(state: RouterState):
    """
    FINAL summarization node.
    MUST NOT re-derive or repeat raw logs.
    Only compress structured output into human report.
    """

    prompt = f"""
You are a senior DevOps engineer writing a FINAL incident report.

IMPORTANT RULES:
- Do NOT repeat logs or raw diagnostic output.
- Do NOT restate Kubernetes outputs or tool results.
- Only summarize the key findings.

Incident:
{state["incident"]}

Key Findings (already analyzed):
{state["diagnosis"]}

Commands (for engineers):
{state["commands"]}

Write output in this format:

Summary:
<2-3 lines max>

Root Cause:
<1-2 lines max>

Action Plan:
<bullet points only>
"""

    report = llm.invoke(prompt).content

    print("\n================ INCIDENT REPORT ================\n")
    print(report)
    print("\n=================================================\n")

    return {
        **state,
        "final_response": report
    }

# =========================================================
# 7. ROUTING FUNCTION (LangGraph conditional edges)
# =========================================================

def route_selector(state: RouterState) -> Literal[
    "kubernetes",
    "database",
    "network",
    "cicd"
]:
    return state["route"]


# =========================================================
# 8. BUILD GRAPH
# =========================================================

graph = StateGraph(RouterState)

# Nodes
graph.add_node("router", router_node)
graph.add_node("kubernetes", kubernetes_agent)
graph.add_node("database", database_agent)
graph.add_node("network", network_agent)
graph.add_node("cicd", cicd_agent)
graph.add_node("response", response_node)

# Entry
graph.set_entry_point("router")

# Conditional routing from LLM decision
graph.add_conditional_edges(
    "router",
    route_selector
)

# Specialist → final response
graph.add_edge("kubernetes", "response")
graph.add_edge("database", "response")
graph.add_edge("network", "response")
graph.add_edge("cicd", "response")

# End
graph.add_edge("response", END)

# Compile workflow
workflow = graph.compile()