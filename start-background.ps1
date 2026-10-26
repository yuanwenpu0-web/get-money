$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$taskName = 'QuantDesk-GetMoney'
$python = Join-Path $projectRoot '.venv\Scripts\pythonw.exe'
$runner = Join-Path $projectRoot 'scripts\background_server.py'

if (-not (Test-Path -LiteralPath $python)) {
    throw 'Project Python environment is missing.'
}
if (-not (Test-Path -LiteralPath (Join-Path $projectRoot 'frontend\dist\index.html'))) {
    throw 'Build the frontend before starting the background server.'
}

$existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existing -and $existing.State -eq 'Running') {
    Write-Output 'Server task is already running: http://127.0.0.1:8010'
    exit 0
}
if (Get-NetTCPConnection -LocalPort 8010 -State Listen -ErrorAction SilentlyContinue) {
    throw 'Port 8010 is already in use. Stop the existing server first.'
}

$action = New-ScheduledTaskAction -Execute $python -Argument ('"{0}"' -f $runner) -WorkingDirectory $projectRoot
$principal = New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew -Hidden
Register-ScheduledTask -TaskName $taskName -Action $action -Principal $principal -Settings $settings -Description 'Quant Desk local web server; started on demand, independent of ChatGPT.' -Force | Out-Null
Start-ScheduledTask -TaskName $taskName
Write-Output 'Background server started: http://127.0.0.1:8010'
Write-Output ('Log: ' + (Join-Path $projectRoot 'output\server.log'))
