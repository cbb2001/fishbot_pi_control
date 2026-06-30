[CmdletBinding()]
param(
    [string]$PiIp = "",
    [string]$Subnet = "192.168.1.0/24",
    [string]$RemoteUser = "fish",
    [int]$ConnectTimeoutSeconds = 5
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Get-NmapScanIps {
    param([Parameter(Mandatory = $true)][string]$ScanSubnet)

    $nmapCommand = Get-Command nmap -ErrorAction SilentlyContinue
    if (-not $nmapCommand) {
        throw "nmap was not found. Install nmap or pass -PiIp to skip scanning."
    }

    Write-Host "Scanning subnet with nmap: $ScanSubnet"
    try {
        $scanOutput = & $nmapCommand.Source -sn $ScanSubnet 2>&1
        $exitCode = $LASTEXITCODE
    }
    catch {
        throw "nmap could not be started. Try running PowerShell as Administrator, or pass -PiIp. Details: $($_.Exception.Message)"
    }

    if ($exitCode -ne 0) {
        $text = ($scanOutput | ForEach-Object { $_.ToString() }) -join "`n"
        throw "nmap scan failed with exit code $exitCode. Try running PowerShell as Administrator, or pass -PiIp. Output:`n$text"
    }

    $ipPattern = "(?:\d{1,3}\.){3}\d{1,3}"
    $ips = foreach ($line in $scanOutput) {
        $text = $line.ToString().Trim()
        if ($text -match "^Nmap scan report for\s+(.+)$") {
            $target = $Matches[1].Trim()
            if ($target -match "\(($ipPattern)\)") {
                $Matches[1]
            }
            elseif ($target -match "^\s*($ipPattern)\s*$") {
                $Matches[1]
            }
        }
    }

    @($ips | Sort-Object -Unique)
}

function Test-PiSsh {
    param(
        [Parameter(Mandatory = $true)][string]$Ip,
        [Parameter(Mandatory = $true)][string]$User,
        [Parameter(Mandatory = $true)][int]$TimeoutSeconds
    )

    $target = "${User}@${Ip}"
    $sshArgs = @(
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=$TimeoutSeconds",
        "-o", "StrictHostKeyChecking=accept-new",
        $target,
        'echo HOSTNAME:$(hostname); echo WHOAMI:$(whoami)'
    )

    try {
        $output = & ssh @sshArgs 2>&1
        $exitCode = $LASTEXITCODE
    }
    catch {
        return [pscustomobject]@{
            Ip       = $Ip
            Ok       = $false
            Hostname = ""
            Whoami   = ""
            Error    = $_.Exception.Message
        }
    }

    if ($exitCode -ne 0) {
        return [pscustomobject]@{
            Ip       = $Ip
            Ok       = $false
            Hostname = ""
            Whoami   = ""
            Error    = (($output | ForEach-Object { $_.ToString() }) -join "`n")
        }
    }

    $hostnameLine = $output | Where-Object { $_ -like "HOSTNAME:*" } | Select-Object -First 1
    $whoamiLine = $output | Where-Object { $_ -like "WHOAMI:*" } | Select-Object -First 1

    return [pscustomobject]@{
        Ip       = $Ip
        Ok       = $true
        Hostname = (($hostnameLine -replace "^HOSTNAME:", "").Trim())
        Whoami   = (($whoamiLine -replace "^WHOAMI:", "").Trim())
        Error    = ""
    }
}

if ($PiIp.Trim().Length -gt 0) {
    $result = Test-PiSsh -Ip $PiIp.Trim() -User $RemoteUser -TimeoutSeconds $ConnectTimeoutSeconds
    if (-not $result.Ok) {
        throw "Could not SSH to ${RemoteUser}@${PiIp} with BatchMode=yes. Check the IP, SSH key, or network permissions. $($result.Error)"
    }

    Write-Host "Using specified Raspberry Pi IP: $($result.Ip)"
    Write-Output $result.Ip
    return
}

$candidateIps = @(Get-NmapScanIps -ScanSubnet $Subnet)
if ($candidateIps.Count -eq 0) {
    throw "No hosts were found by nmap on $Subnet. Try running PowerShell as Administrator, or pass -PiIp."
}

Write-Host "Online hosts found: $($candidateIps -join ', ')"

$sshResults = foreach ($ip in $candidateIps) {
    Write-Host "Checking SSH key login: ${RemoteUser}@${ip}"
    Test-PiSsh -Ip $ip -User $RemoteUser -TimeoutSeconds $ConnectTimeoutSeconds
}

$loginMatches = @($sshResults | Where-Object { $_.Ok -and $_.Whoami -eq $RemoteUser })
$hostnameMatches = @($loginMatches | Where-Object { $_.Hostname -eq "fish" })

if ($hostnameMatches.Count -eq 1) {
    Write-Host "Selected Raspberry Pi by hostname: $($hostnameMatches[0].Ip)"
    Write-Output $hostnameMatches[0].Ip
    return
}

if ($hostnameMatches.Count -gt 1) {
    $ips = ($hostnameMatches | ForEach-Object { $_.Ip }) -join ", "
    throw "Multiple SSH targets have hostname 'fish': $ips. Pass -PiIp to choose one."
}

if ($loginMatches.Count -eq 1) {
    Write-Host "Selected Raspberry Pi by SSH login: $($loginMatches[0].Ip)"
    Write-Output $loginMatches[0].Ip
    return
}

if ($loginMatches.Count -gt 1) {
    $ips = ($loginMatches | ForEach-Object { "$($_.Ip) hostname=$($_.Hostname)" }) -join ", "
    throw "Multiple hosts accept SSH login as '$RemoteUser': $ips. Pass -PiIp to choose one."
}

throw "No scanned host accepted SSH key login as '$RemoteUser'. Check that the Pi is online and the SSH key is installed, or pass -PiIp."
