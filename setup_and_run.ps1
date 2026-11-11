# One-shot: create Python env, install CUDA PyTorch, download + prepare datasets, train on the GPU.
# Run via START_HERE.bat (double-click) or:  powershell -ExecutionPolicy Bypass -File setup_and_run.ps1 [pipeline args]
# Extra args go to run_pipeline.py, e.g.:  ... -File setup_and_run.ps1 --no-openbmi
$ErrorActionPreference = 'Continue'   # pip writes warnings to stderr; failures are checked via $LASTEXITCODE
Set-Location $PSScriptRoot
$Venv = Join-Path $PSScriptRoot '.venv'
$Py = Join-Path $Venv 'Scripts\python.exe'

function Find-Python {
    foreach ($v in '3.12', '3.11', '3.10') {
        $p = & py "-$v" -c "import sys;print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $p) { return $p }
    }
    return $null
}

if (-not (Test-Path $Py)) {
    $sys = if (Get-Command py -ErrorAction SilentlyContinue) { Find-Python } else { $null }
    if (-not $sys) {
        Write-Host 'Python 3.10-3.12 not found. Installing Python 3.12 with winget...'
        winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements
        $env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + [Environment]::GetEnvironmentVariable('Path', 'User')
        $sys = if (Get-Command py -ErrorAction SilentlyContinue) { Find-Python } else { $null }
        if (-not $sys) { throw 'Python install did not finish. Install Python 3.12 from python.org (tick "Add to PATH"), then re-run.' }
    }
    Write-Host "Creating virtual environment with $sys"
    & $sys -m venv $Venv
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the virtual environment' }
}

if (-not (Test-Path (Join-Path $Venv '.deps_ok'))) {
    & $Py -m pip install --upgrade pip
    if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) { Write-Warning 'nvidia-smi not found: install the NVIDIA driver first or training will not use the GPU.' }
    $ok = $false
    foreach ($cu in 'cu126', 'cu124') {   # newest CUDA wheel first; falls back for older drivers
        Write-Host "Installing PyTorch ($cu)..."
        & $Py -m pip install torch --index-url "https://download.pytorch.org/whl/$cu"
        if ($LASTEXITCODE -eq 0) {
            & $Py -c "import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)"
            if ($LASTEXITCODE -eq 0) { $ok = $true; break }
            Write-Warning "torch ($cu) installed but CUDA not available (driver too old?). Trying the next build."
        }
    }
    if (-not $ok) { throw 'No working CUDA build of PyTorch. Update the NVIDIA driver (https://www.nvidia.com/drivers) and re-run.' }
    & $Py -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw 'pip install -r requirements.txt failed' }
    New-Item -ItemType File -Path (Join-Path $Venv '.deps_ok') | Out-Null
}

& $Py run_pipeline.py @args
exit $LASTEXITCODE
