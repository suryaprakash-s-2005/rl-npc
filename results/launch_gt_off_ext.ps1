$ErrorActionPreference = "Continue"
$args = @('-u','experiments/run_gt_off_ext.py')
$p = Start-Process -FilePath "python" -ArgumentList $args -WorkingDirectory "D:\IEEE\project" -RedirectStandardOutput "D:\IEEE\project\results\gt_off_ext_log.txt" -RedirectStandardError "D:\IEEE\project\results\gt_off_ext_log.err" -PassThru
Write-Output ("PID=" + $p.Id)
