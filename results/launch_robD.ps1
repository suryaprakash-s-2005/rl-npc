$ErrorActionPreference = "Continue"
Remove-Item -Force "D:\IEEE\project\results\train_robD.txt","D:\IEEE\project\results\train_robD.err" -ErrorAction SilentlyContinue
$args = @('-u','-m','src.train','--rl.replay','partitioned','--rl.strategy_source','groundtruth','--seed','2','--rl.episodes','600','--rl.seed_demos','80','--rl.params_jitter','0.18','--rl.epsilon_start','0.15','--rl.epsilon_end','0.02','--rl.epsilon_decay_steps','12000')
$p = Start-Process -FilePath "python" -ArgumentList $args -WorkingDirectory "D:\IEEE\project" -RedirectStandardOutput "D:\IEEE\project\results\train_robD.txt" -RedirectStandardError "D:\IEEE\project\results\train_robD.err" -PassThru
Write-Output ("PID=" + $p.Id)
