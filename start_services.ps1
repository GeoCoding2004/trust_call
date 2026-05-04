# start_services.ps1
Write-Host "🚀 Starting Trust-Call Microservices Architecture..." -ForegroundColor Cyan

# Helper function to reliably activate the virtual environment
function Activate-Venv {
    if (Test-Path ".\venv\Scripts\Activate.ps1") {
        .\venv\Scripts\Activate.ps1
    } elseif (Test-Path ".\.venv\Scripts\Activate.ps1") {
        .\.venv\Scripts\Activate.ps1
    } else {
        Write-Host "⚠️ Warning: No virtual environment found in $(Get-Location)" -ForegroundColor Yellow
    }
}

# 1. Start RawNet Service (Port 8000)
Start-Process powershell -ArgumentList "-NoExit", "-Command", "& { cd E:\trust_call\rawnet-service; function Activate-Venv { if (Test-Path '.\venv\Scripts\Activate.ps1') { .\venv\Scripts\Activate.ps1 } elseif (Test-Path '.\.venv\Scripts\Activate.ps1') { .\.venv\Scripts\Activate.ps1 } }; Activate-Venv; Write-Host '--- RAWNET SERVICE (Port 8000) ---' -ForegroundColor Magenta; uvicorn main:app --host 0.0.0.0 --port 8000 }"

# 2. Start DistilBERT Service (Port 8002)
Start-Process powershell -ArgumentList "-NoExit", "-Command", "& { cd E:\trust_call\distilbert-service; function Activate-Venv { if (Test-Path '.\venv\Scripts\Activate.ps1') { .\venv\Scripts\Activate.ps1 } elseif (Test-Path '.\.venv\Scripts\Activate.ps1') { .\.venv\Scripts\Activate.ps1 } }; Activate-Venv; Write-Host '--- DISTILBERT SERVICE (Port 8002) ---' -ForegroundColor Magenta; uvicorn main:app --host 0.0.0.0 --port 8002 }"

# Wait 3 seconds to let the AI models load into RAM (Warm Start)
Write-Host "⏳ Waiting for AI models to boot into RAM..." -ForegroundColor Yellow
Start-Sleep -Seconds 3

# 3. Start The Brain / EEP Gateway (Port 8080)
Start-Process powershell -ArgumentList "-NoExit", "-Command", "& { cd E:\trust_call\trust_call_backend; function Activate-Venv { if (Test-Path '.\venv\Scripts\Activate.ps1') { .\venv\Scripts\Activate.ps1 } elseif (Test-Path '.\.venv\Scripts\Activate.ps1') { .\.venv\Scripts\Activate.ps1 } }; Activate-Venv; Write-Host '--- EEP GATEWAY & WHISPER (Port 8080) ---' -ForegroundColor Green; python server.py }"

# 4. Start React Native Metro Bundler
Start-Process powershell -ArgumentList "-NoExit", "-Command", "& { cd E:\trust_call\TrustCallApp; Write-Host '--- METRO BUNDLER ---' -ForegroundColor Cyan; npm start }"

Write-Host "✅ All 4 services launched!" -ForegroundColor Green