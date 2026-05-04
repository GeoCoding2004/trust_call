# Kubernetes Deployment

This manifest provides the required Kubernetes shape for Trust-Call:

- `rawnet-service`: IEP1 synthetic voice detection
- `distilbert-service`: IEP2 semantic scam/coercion detection
- `trust-call-backend`: public EEP / WebRTC gateway / late fusion boundary
- `prometheus`: metrics collection
- `grafana`: dashboard UI

Before applying it to a real cluster, replace the placeholder image names:

```text
ghcr.io/YOUR_ORG/trust-call-rawnet:latest
ghcr.io/YOUR_ORG/trust-call-distilbert:latest
ghcr.io/YOUR_ORG/trust-call-backend:latest
```

Then apply:

```powershell
kubectl apply -f deployment/kubernetes/trust-call-stack.yaml
kubectl get pods -n trust-call
kubectl get services -n trust-call
```

For the Azure deadline demo, Azure Container Apps is the faster runtime target.
These manifests document and enable the Kubernetes deployment path required by
the project rubric.
