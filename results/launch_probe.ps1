$ErrorActionPreference = "Continue"
$args = @('-u','-m','experiments.run_all','--seeds','0','--variants','single,partitioned','--tagging','gt','--cond','on','--episodes','200','--eval_episodes','4')
$p = Start-Process -FilePath "python" -ArgumentList $args -WorkingDirectory "D:\IEEE\project" -RedirectStandardOutput "D:\IEEE\project\results\probe_gt_on.txt" -RedirectStandardError "D:\IEEE\project\results\probe_gt_on.err" -PassThru
Write-Output ("PID=" + $p.Id)
