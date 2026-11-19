$ErrorActionPreference = "Continue"
$args = @('-u','experiments/run_det_ext.py')
$p = Start-Process -FilePath "python" -ArgumentList $args -WorkingDirectory "D:\IEEE\project" -RedirectStandardOutput "D:\IEEE\project\results\det_ext_log.txt" -RedirectStandardError "D:\IEEE\project\results\det_ext_log.err" -PassThru
Write-Output ("PID=" + $p.Id)
