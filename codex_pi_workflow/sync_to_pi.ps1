[CmdletBinding()]
param(
    [string]$PiIp = "",
    [string]$Subnet = "192.168.1.0/24",
    [string]$RemoteUser = "fish",
    [string]$RemoteDir = "/home/fish/fishbot_pi_control"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Invoke-Native {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [int[]]$SuccessCodes = @(0)
    )

    & $FilePath @Arguments
    $exitCode = $LASTEXITCODE
    if ($SuccessCodes -notcontains $exitCode) {
        throw "$FilePath failed with exit code $exitCode"
    }
}

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

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
$expectedRemoteDir = "/home/fish/fishbot_pi_control"
$resolvedPiIp = Resolve-PiIp
$RemoteHost = "${RemoteUser}@${resolvedPiIp}"
$sshOptions = @("-o", "BatchMode=yes", "-o", "ConnectTimeout=8")

if ($RemoteDir -ne $expectedRemoteDir) {
    Write-Warning "RemoteDir is '$RemoteDir'; expected '$expectedRemoteDir'. This script will not delete remote files."
}

$itemsToSync = @(
    "main.py",
    "requirements.txt",
    "config",
    "drivers",
    "control",
    "scripts"
)

$excludedDirs = @(".git", ".venv", "__pycache__", ".vscode", "codex_pi_workflow")
$excludedFiles = @("*.pyc")
$stageDir = Join-Path ([System.IO.Path]::GetTempPath()) ("fishbot_pi_control_sync_" + [System.Guid]::NewGuid().ToString("N"))
$tempRoot = [System.IO.Path]::GetFullPath([System.IO.Path]::GetTempPath())
$stageDirFull = [System.IO.Path]::GetFullPath($stageDir)

if (-not $stageDirFull.StartsWith($tempRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to use staging directory outside temp root: $stageDirFull"
}

Write-Host "Project root: $projectRoot"
Write-Host "Remote target: ${RemoteHost}:$RemoteDir"

New-Item -ItemType Directory -Path $stageDir -Force | Out-Null

try {
    foreach ($item in $itemsToSync) {
        $source = Join-Path $projectRoot $item
        $destination = Join-Path $stageDir $item

        if (-not (Test-Path -LiteralPath $source)) {
            Write-Warning "Skip missing item: $item"
            continue
        }

        if (Test-Path -LiteralPath $source -PathType Container) {
            New-Item -ItemType Directory -Path $destination -Force | Out-Null
            $robocopyArgs = @($source, $destination, "/E", "/XD") + $excludedDirs + @("/XF") + $excludedFiles
            & robocopy @robocopyArgs | Out-Host
            $robocopyExit = $LASTEXITCODE
            if ($robocopyExit -gt 7) {
                throw "robocopy failed while staging '$item' with exit code $robocopyExit"
            }
        }
        else {
            Copy-Item -LiteralPath $source -Destination $destination -Force
        }
    }

    $stagedItems = @(Get-ChildItem -LiteralPath $stageDir -Force)
    if ($stagedItems.Count -eq 0) {
        Write-Warning "Nothing to sync. No expected project files or directories were found."
        return
    }

    Write-Host "Creating remote directory..."
    Invoke-Native -FilePath "ssh" -Arguments ($sshOptions + @($RemoteHost, "mkdir -p $(Quote-Bash $RemoteDir)"))

    if ($RemoteDir -eq $expectedRemoteDir) {
        Write-Host "Mirroring staged items inside safe remote project directory..."
        foreach ($entry in $stagedItems) {
            $remoteItem = "$RemoteDir/$($entry.Name)"
            Invoke-Native -FilePath "ssh" -Arguments ($sshOptions + @($RemoteHost, "rm -rf -- $(Quote-Bash $remoteItem)"))
        }
    }
    else {
        Write-Warning "RemoteDir is not the expected project path; skipping remote cleanup."
    }

    foreach ($entry in $stagedItems) {
        Write-Host "Copying $($entry.Name)..."
        Invoke-Native -FilePath "scp" -Arguments ($sshOptions + @("-r", $entry.FullName, "${RemoteHost}:$RemoteDir/"))
    }

    Write-Host "Ensuring remote captures directory exists..."
    Invoke-Native -FilePath "ssh" -Arguments ($sshOptions + @($RemoteHost, "mkdir -p $(Quote-Bash "$RemoteDir/captures")"))

    Write-Host ""
    Write-Host "Sync complete."
    Write-Host "Next:"
    Write-Host "  .\codex_pi_workflow\setup_pi_venv.ps1"
    Write-Host "  .\codex_pi_workflow\run_on_pi.ps1 scripts/test_i2c.py"
}
finally {
    if ((Test-Path -LiteralPath $stageDirFull) -and $stageDirFull.StartsWith($tempRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        Remove-Item -LiteralPath $stageDirFull -Recurse -Force
    }
}
