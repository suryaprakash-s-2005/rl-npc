$ErrorActionPreference = "Continue"
$args = @('-u','-m','experiments.run_all','--seeds','0','--variants','single,partitioned','--tagging','gt','--cond','on','--episodes','400','--eval_episodes','10')
$p = Start-Process -FilePath "python" -ArgumentList $args -WorkingDirectory "D:\IEEE\project" -RedirectStandardOutput "D:\IEEE\project\results\val_sp.txt" -RedirectStandardError "D:\IEEE\project\results\val_sp.err" -PassThru
Write-Output ("PID=" + $p.Id)
