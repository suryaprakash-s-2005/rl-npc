$ErrorActionPreference = "Continue"
$args = @('-u','experiments/run_grid.py')
$p = Start-Process -FilePath "python" -ArgumentList $args -WorkingDirectory "D:\IEEE\project" -RedirectStandardOutput "D:\IEEE\project\results\grid_log.txt" -RedirectStandardError "D:\IEEE\project\results\grid_log.err" -PassThru
Write-Output ("PID=" + $p.Id)
