[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$Script = "main.py",

    [Parameter(Position = 1, ValueFromRemainingArguments = $true)]
    [string[]]$ScriptArgs = @(),

    [string]$PiIp = "",
    [string]$Subnet = "192.168.1.0/24",
    [string]$RemoteUser = "fish",
    [string]$RemoteDir = "/home/fish/fishbot_pi_control",
    [switch]$NoVenv
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

$pythonArgs = @($Script) + $ScriptArgs
$quotedPythonArgs = ($pythonArgs | ForEach-Object { Quote-Bash $_ }) -join " "
if ($NoVenv) {
    $remoteCommand = "cd $(Quote-Bash $RemoteDir) && python3 $quotedPythonArgs"
}
else {
    $remoteCommand = "cd $(Quote-Bash $RemoteDir) && source .venv/bin/activate && python3 $quotedPythonArgs"
}
$sshArgs = $sshOptions + @($RemoteHost, $remoteCommand)

Write-Host "Running on Raspberry Pi:"
Write-Host "  ssh $RemoteHost `"$remoteCommand`""

& ssh @sshArgs
$exitCode = $LASTEXITCODE
if ($exitCode -ne 0) {
    throw "Remote command failed with exit code $exitCode"
}
