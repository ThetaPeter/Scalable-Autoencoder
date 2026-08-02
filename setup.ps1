$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

Write-Host "Creating Python 3.11 virtual environment..."
py -3.11 -m venv .venv

Write-Host "Activating virtual environment..."
& ".\.venv\Scripts\Activate.ps1"

$pipTrustedHosts = @(
    "pypi.org",
    "files.pythonhosted.org",
    "download.pytorch.org"
)
$pipTrustArgs = $pipTrustedHosts | ForEach-Object { "--trusted-host"; $_ }

Write-Host "Upgrading pip..."
python -m pip install @pipTrustArgs --upgrade pip

Write-Host "Installing dependencies..."
pip install @pipTrustArgs -r requirements.txt

Write-Host "Running CUDA smoke test..."
python -c "import torch; print('PyTorch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"

Write-Host ""
Write-Host "Setup complete. Activate with: .\.venv\Scripts\Activate.ps1"
Write-Host "Run training with:"
Write-Host '  $env:PYTHONUTF8="1"; python train_autoencoder.py'
