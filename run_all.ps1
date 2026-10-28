# ==============================================================================
# Background Noise Suppression (DeepFilterNet) - Reproducible End-to-End Runner
# Target: Windows 10/11 x64, Python 3.11, PyTorch 2.0.1+CPU
# ==============================================================================

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $ScriptDir

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host " [1/5] Checking Environment and Python Runtime" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

$PythonExe = "$ScriptDir\venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    Write-Warning "Local venv python not found at $PythonExe. Falling back to system 'python'."
    $PythonExe = "python"
}

& $PythonExe verify_env.py
if ($LASTEXITCODE -ne 0) {
    Write-Error "verify_env.py failed. Please check dependencies in requirements.txt."
    exit 1
}

if (-not (Test-Path "$ScriptDir\outputs")) {
    New-Item -ItemType Directory -Path "$ScriptDir\outputs" | Out-Null
}

Write-Host "`n======================================================================" -ForegroundColor Cyan
Write-Host " [2/5] Testing Offline Enhancement Pipeline (src/denoise.py)" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

$SampleFile = "$ScriptDir\samples\synthetic_noisy_speech.wav"
if (-not (Test-Path $SampleFile)) {
    $SampleFile = "$ScriptDir\samples\clean_speech_ref.wav"
}

$OfflineOut = "$ScriptDir\outputs\run_all_offline_test.wav"
Write-Host "Running denoise.py on '$SampleFile' -> '$OfflineOut'..." -ForegroundColor Yellow
& $PythonExe src\denoise.py "$SampleFile" -o "$OfflineOut"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Offline denoiser failed!"
    exit 1
}
Write-Host "Offline test successful: $OfflineOut generated." -ForegroundColor Green

Write-Host "`n======================================================================" -ForegroundColor Cyan
Write-Host " [3/5] Testing Real-Time WebSocket Streaming Pipeline" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

# Check if server is already running on port 8000
$PortOpen = $false
try {
    $Tcp = New-Object System.Net.Sockets.TcpClient
    $Connect = $Tcp.BeginConnect("127.0.0.1", 8000, $null, $null)
    $Wait = $Connect.AsyncWaitHandle.WaitOne(800, $false)
    if ($Wait -and $Tcp.Connected) {
        $Tcp.EndConnect($Connect)
        $PortOpen = $true
    }
    $Tcp.Close()
} catch {
    $PortOpen = $false
}

$ServerJob = $null
if (-not $PortOpen) {
    Write-Host "Starting background realtime server on port 8000..." -ForegroundColor Yellow
    $ServerJob = Start-Process -FilePath $PythonExe -ArgumentList "src\realtime\server.py --port 8000" -PassThru -WindowStyle Hidden
    Start-Sleep -Seconds 3
} else {
    Write-Host "Realtime server already detected running on port 8000." -ForegroundColor Green
}

$StreamOut = "$ScriptDir\outputs\run_all_stream_test.wav"
Write-Host "Streaming '$SampleFile' through WebSocket client -> '$StreamOut'..." -ForegroundColor Yellow
& $PythonExe src\realtime\client.py --file "$SampleFile" --output "$StreamOut" --port 8000
$StreamResult = $LASTEXITCODE

if ($ServerJob -ne $null) {
    Write-Host "Stopping temporary background server process..." -ForegroundColor Gray
    Stop-Process -Id $ServerJob.Id -Force -ErrorAction SilentlyContinue
}

if ($StreamResult -ne 0) {
    Write-Error "Realtime streaming test failed!"
    exit 1
}
Write-Host "Real-time stream test successful: $StreamOut generated." -ForegroundColor Green

Write-Host "`n======================================================================" -ForegroundColor Cyan
Write-Host " [4/5] Running Evaluation and Metric Benchmark Suite (src/evaluate.py)" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

& $PythonExe src\evaluate.py
if ($LASTEXITCODE -ne 0) {
    Write-Error "evaluate.py failed!"
    exit 1
}

Write-Host "`n======================================================================" -ForegroundColor Cyan
Write-Host " [5/5] Verification Complete!" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "All pipelines executed cleanly." -ForegroundColor Green
Write-Host "Evaluation results written to:" -ForegroundColor Green
Write-Host "  - outputs\evaluation_results.md" -ForegroundColor Gray
Write-Host "  - outputs\evaluation_results.csv" -ForegroundColor Gray
Write-Host "  - outputs\sanity_checks.md" -ForegroundColor Gray
Write-Host "`nTo launch the interactive Web UI:" -ForegroundColor Yellow
Write-Host "  .\venv\Scripts\python.exe src\app.py" -ForegroundColor White
Write-Host "To start the persistent Real-time WebSocket server:" -ForegroundColor Yellow
Write-Host "  .\venv\Scripts\python.exe src\realtime\server.py --port 8000" -ForegroundColor White
Write-Host "To stream live microphone in real time:" -ForegroundColor Yellow
Write-Host "  .\venv\Scripts\python.exe src\realtime\client.py" -ForegroundColor White
Write-Host "To launch the Live Voice Call Studio (Web UI):" -ForegroundColor Yellow
Write-Host "  .\venv\Scripts\python.exe src\call\call_server.py --port 8001" -ForegroundColor White

