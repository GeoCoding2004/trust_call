# Azure Deployment: IEP1 and IEP2

This deployment keeps IEP3 private/local. Only these services are intended for Azure:

- `rawnet-service`: IEP1 synthetic/deepfake voice scoring
- `distilbert-service`: IEP2 semantic scam/coercion scoring

The local Trust-Call gateway should point to the deployed service URLs with:

```powershell
$env:RAWNET_URL="https://<rawnet-app-url>/predict"
$env:DISTILBERT_URL="https://<distilbert-app-url>/predict"
```

## Recommended Azure Shape

Use Azure Container Apps for the demo:

- Azure Container Registry: Basic
- RawNet container app: 2 vCPU / 4 GiB memory, min replicas 1 for demo
- DistilBERT container app: 2 vCPU / 4 GiB memory, min replicas 1 for demo

For privacy, do not enable request body logging. The services process audio/text in memory and expose only predictions plus `/metrics`.

## Build Images Locally

Run from the repo root:

```powershell
$ACR_NAME="<your-acr-name>"
$RAWNET_IMAGE="$ACR_NAME.azurecr.io/trust-call-rawnet:latest"
$DISTILBERT_IMAGE="$ACR_NAME.azurecr.io/trust-call-distilbert:latest"

az acr login --name $ACR_NAME

docker build -t $RAWNET_IMAGE .\rawnet-service
docker build -t $DISTILBERT_IMAGE .\distilbert-service

docker push $RAWNET_IMAGE
docker push $DISTILBERT_IMAGE
```

## Local Backend After Deployment

Restart the local gateway with the Azure endpoints:

```powershell
cd C:\Users\JL\Desktop\trust_call_full_pipeline
$env:RAWNET_URL="https://<rawnet-app-url>/predict"
$env:DISTILBERT_URL="https://<distilbert-app-url>/predict"
& "C:\Users\JL\Desktop\trust_call\.venv\Scripts\python.exe" -m uvicorn trust_call_backend.server:app --host 0.0.0.0 --port 8080
```

The mobile app still talks to the local gateway during this phase. IEP3 remains local to that gateway and is not deployed to Azure.
