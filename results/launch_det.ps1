$ErrorActionPreference = "Continue"
$log = "D:\IEEE\project\results\grid_det.txt"
$err = "D:\IEEE\project\results\grid_det.err"
Remove-Item -Force $log, $err -ErrorAction SilentlyContinue
$p = Start-Process -FilePath "D:\venv\Scripts\python.exe" `
    -ArgumentList '-u','-m','experiments.run_all','--seeds','0,2,4','--variants','single,partitioned,partitioned_noprior','--tagging','det','--cond','on','--episodes','400','--eval_episodes','12' `
    -WorkingDirectory "D:\IEEE\project" `
    -RedirectStandardOutput $log -RedirectStandardError $err `
    -PassThru
Write-Output ("PID=" + $p.Id)
