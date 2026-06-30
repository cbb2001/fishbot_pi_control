[CmdletBinding()]
param(
    [string]$PiIp = "",
    [string]$Subnet = "192.168.1.0/24",
    [string]$RemoteUser = "fish",
    [string]$RemoteDir = "/home/fish/fishbot_pi_control"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$scriptDir = Split-Path -Parent $PSCommandPath

function Quote-Bash {
    param([Parameter(Mandatory = $true)][string]$Text)
    $singleQuote = [char]39
    return $singleQuote + $Text.Replace("$singleQuote", "$singleQuote\$singleQuote$singleQuote") + $singleQuote
}

function Resolve-PiIp {
    $findPiScript = Join-Path $scriptDir "find_pi.ps1"
    if (-not (Test-Path -LiteralPath $findPiScript)) {
        throw "Missing Raspberry Pi discovery script: $findPiScript"
    }

    $findArgs = @{
        Subnet     = $Subnet
        RemoteUser = $RemoteUser
    }
    if ($PiIp.Trim().Length -gt 0) {
        $findArgs.PiIp = $PiIp.Trim()
    }

    $resolved = & $findPiScript @findArgs | Select-Object -Last 1
    if (-not $resolved) {
        throw "Could not resolve Raspberry Pi IP."
    }

    return $resolved.ToString().Trim()
}

$resolvedPiIp = Resolve-PiIp
$RemoteHost = "${RemoteUser}@${resolvedPiIp}"
$sshOptions = @("-o", "BatchMode=yes", "-o", "ConnectTimeout=8")

$remoteCommand = @"
mkdir -p $(Quote-Bash $RemoteDir) &&
cd $(Quote-Bash $RemoteDir) &&
if [ ! -d .venv ]; then python3 -m venv .venv; fi &&
source .venv/bin/activate &&
if [ -f requirements.txt ]; then python3 -m pip install -r requirements.txt; else echo 'requirements.txt not found; skipping pip install'; fi
"@ -replace "`r?`n", " "

$sshArgs = $sshOptions + @($RemoteHost, $remoteCommand)

Write-Host "Setting up Raspberry Pi virtual environment:"
Write-Host "  ssh $RemoteHost `"$remoteCommand`""

& ssh @sshArgs
$exitCode = $LASTEXITCODE
if ($exitCode -ne 0) {
    throw "Remote setup failed with exit code $exitCode"
}
