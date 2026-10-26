$ErrorActionPreference = 'Stop'
$taskName = 'QuantDesk-GetMoney'
$task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($task -and $task.State -eq 'Running') {
    Stop-ScheduledTask -TaskName $taskName
    Write-Output 'Background server stopped.'
} else {
    Write-Output 'Background server is not running.'
}
