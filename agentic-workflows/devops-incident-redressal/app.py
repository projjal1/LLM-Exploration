# DevOps Incident AI Agent using LangGraph and Qwen
# LLM-based router selects the appropriate specialist agent, which uses domain-specific tools to diagnose the issue 
# and produce actionable commands, followed by a final summarized incident report.

# Knowledge domains are: database, kubernetes, network, and CI/CD.

from workflow import workflow

# Short banner for the CLI
def print_banner():
    print("\n" + "=" * 60)
    print("🚀 DEVOPS INCIDENT AI AGENT (LOCAL QWEN + LANGGRAPH)")
    print("=" * 60)
    print("Type an incident description.")
    print("Type 'exit' to quit.")
    print("=" * 60 + "\n")

# Main loop to run the agent
def run_agent():
    print_banner()

    # Loop to continuously accept incidents from the user
    # Exit when user types 'exit'
    while True:

        # Get user input for incident
        incident = input("\n🧑‍💻 Incident > ")

        # Exit condition
        if incident.lower().strip() == "exit":
            print("\n👋 Shutting down agent...\n")
            break

        # Check for empty input
        if not incident.strip():
            print("Please enter a valid incident.")
            continue

        # Initial state passed into LangGraph
        initial_state = {
            "incident": incident,
            "route": None,
            "diagnosis": None,
            "commands": None,
            "evidence": None,
            "final_response": None
        }

        # Invoke workflow
        result = workflow.invoke(initial_state)

        # Print final output safely (in case response_node didn't print)
        print("\n\n📦 FINAL OUTPUT STORED IN STATE\n")
        print("Route:", result.get("route"))
        print("\n--- Diagnosis ---\n")
        print(result.get("diagnosis"))

        print("\n--- Commands ---\n")
        commands = result.get("commands") or []
        for cmd in commands:
            print("👉", cmd)

        print("\n--- Final Response ---\n")
        print(result.get("final_response"))


if __name__ == "__main__":
    run_agent()

# List of incidents for testing
'''
incidents = [
    "Payment pods keep restarting after deployment",
    "Database connection timeout errors in production",
    "Network latency spikes causing service degradation",
    "CI/CD pipeline fails to deploy new version"
]
'''

'''
Output for incident #1: "Payment pods keep restarting after deployment"

🚢 Kubernetes Agent running...


================ INCIDENT REPORT ================

Summary:  
Payment-service pods (3 total, 1 healthy) are restarting 14 times, leading to degraded status.  

Root Cause:  
CrashLoopBackOff due to application errors or misconfiguration in the deployment.  

Action Plan:  
- Review pod logs for specific error messages.  
- Check resource limits (CPU/Memory) and adjust if constrained.  
- Validate deployment configuration for syntax or environment variable issues.  
- Investigate recent events for failed container creation or startup errors.  
- Ensure service dependencies (e.g., databases, APIs) are reachable and stable.

=================================================



📦 FINAL OUTPUT STORED IN STATE

Route: kubernetes

--- Diagnosis ---


Kubernetes Issue Detected for payment-service

Findings:

POD STATUS FOR payment-service

Total Pods: 3
Healthy Pods: 1
Restarts: 14

STATUS: degraded



--- Commands ---

👉 kubectl get pods -n production | grep payment-service
👉 kubectl logs deployment/payment-service
👉 kubectl describe pod payment-service-pod-1
👉 kubectl get events --sort-by=.lastTimestamp

--- Final Response ---

Summary:  
Payment-service pods (3 total, 1 healthy) are restarting 14 times, leading to degraded status.  

Root Cause:  
CrashLoopBackOff due to application errors or misconfiguration in the deployment.  

Action Plan:  
- Review pod logs for specific error messages.  
- Check resource limits (CPU/Memory) and adjust if constrained.  
- Validate deployment configuration for syntax or environment variable issues.  
- Investigate recent events for failed container creation or startup errors.  
- Ensure service dependencies (e.g., databases, APIs) are reachable and stable.
'''