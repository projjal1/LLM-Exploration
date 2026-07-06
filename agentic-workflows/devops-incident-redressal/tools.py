from langchain_core.tools import tool
# from langchain.tools import ToolRuntime
import random
import time
from datetime import datetime


# =========================================================
# Simulated Infrastructure State (fake "real-time system")
# =========================================================

SERVICES = {
    "payment-service": {
        "status": "degraded",
        "pods": 3,
        "healthy_pods": 1,
        "restarts": 14
    },
    "auth-service": {
        "status": "healthy",
        "pods": 4,
        "healthy_pods": 4,
        "restarts": 0
    },
    "order-service": {
        "status": "degraded",
        "pods": 5,
        "healthy_pods": 2,
        "restarts": 7
    }
}

DATABASE = {
    "postgres": {
        "connections": 320,
        "max_connections": 500,
        "slow_queries": 12,
        "status": "degraded"
    }
}

CI_PIPELINES = {
    "payment-deploy": {
        "status": "failed",
        "last_run": "12 min ago",
        "error": "Kubernetes rollout timeout"
    },
    "auth-deploy": {
        "status": "success",
        "last_run": "1 hour ago",
        "error": None
    }
}


# =========================================================
# KUBERNETES TOOLS
# =========================================================

@tool
def kubectl_get_pods(service: str) -> str:
    """
    Simulate kubectl get pods for a service.
    """

    svc = SERVICES.get(service, None)

    if not svc:
        return f"Service '{service}' not found in cluster."

    return f"""
POD STATUS FOR {service}

Total Pods: {svc['pods']}
Healthy Pods: {svc['healthy_pods']}
Restarts: {svc['restarts']}

STATUS: {svc['status']}
"""


@tool
def kubectl_describe_pod(service: str) -> str:
    """
    Simulate kubectl describe pod output.
    """

    return f"""
Name: {service}-pod-1
Namespace: production
Status: Running

Events:
- Pulling image...
- Image pulled successfully
- Container started
- Warning: Restart loop detected
"""


@tool
def kubectl_logs(service: str) -> str:
    """
    Simulate kubectl logs output.
    """

    logs = [
        "Starting payment worker...",
        "Connecting to database...",
        "Transaction timeout detected",
        "Retrying connection...",
        "CRITICAL: Pod restart triggered"
    ]

    return "\n".join(logs)


@tool
def kubectl_get_events(cluster: str) -> str:
    """
    Simulate kubernetes cluster events.
    """

    return f"""
EVENTS (last 10 min)

[{datetime.now()}]
- Pod payment-service restarted
- Liveness probe failed
- Node pressure detected
"""


# =========================================================
# DATABASE TOOLS (PostgreSQL-like)
# =========================================================

@tool
def db_connection_stats(_: str) -> str:
    """
    Simulate database connection stats.
    """

    db = DATABASE["postgres"]

    return f"""
DATABASE CONNECTIONS

Active: {db['connections']}
Max: {db['max_connections']}

Usage: {round(db['connections']/db['max_connections']*100, 2)}%
Status: {db['status']}
"""


@tool
def db_slow_queries(_: str) -> str:
    """
    Simulate slow query analysis.
    """

    queries = [
        "SELECT * FROM transactions WHERE user_id = ?",
        "SELECT * FROM payments ORDER BY created_at DESC",
        "UPDATE orders SET status = 'processing'"
    ]

    return f"""
SLOW QUERIES DETECTED: {DATABASE['postgres']['slow_queries']}

Top offenders:
- {queries[0]}
- {queries[1]}
- {queries[2]}
"""


# =========================================================
# NETWORK TOOLS
# =========================================================

@tool
def network_ping(service: str) -> str:
    """
    Simulate network ping test.
    """

    latency = random.randint(20, 250)

    status = "OK" if latency < 120 else "HIGH LATENCY"

    return f"""
PING RESULT: {service}

Latency: {latency}ms
Status: {status}
Packet Loss: {random.choice([0, 0, 1, 2])}%
"""


@tool
def dns_lookup(domain: str) -> str:
    """
    Simulate DNS resolution.
    """

    return f"""
DNS LOOKUP: {domain}

A Record: 192.168.1.{random.randint(10, 200)}
Resolved Successfully: True
TTL: 120s
"""


# =========================================================
# CI/CD TOOLS
# =========================================================

@tool
def ci_pipeline_status(pipeline: str) -> str:
    """
    Simulate CI/CD pipeline status.
    """

    pipe = CI_PIPELINES.get(pipeline, None)

    if not pipe:
        return f"Pipeline '{pipeline}' not found."

    return f"""
PIPELINE: {pipeline}

Status: {pipe['status']}
Last Run: {pipe['last_run']}
Error: {pipe['error']}
"""


@tool
def rollback_deployment(service: str) -> str:
    """
    Simulate rollback command execution.
    """

    return f"""
ROLLBACK INITIATED

Service: {service}
Action: Rolling back to previous stable version
Status: IN PROGRESS

Estimated time: 2-3 minutes
"""