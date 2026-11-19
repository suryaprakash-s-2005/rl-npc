$ErrorActionPreference = "Continue"
Remove-Item -Force "D:\IEEE\project\results\train_robA.txt","D:\IEEE\project\results\train_robA.err" -ErrorAction SilentlyContinue
$args = @('-u','-m','src.train','--rl.replay','partitioned','--rl.strategy_source','groundtruth','--seed','0','--rl.episodes','600','--rl.seed_demos','40','--rl.params_jitter','0.2','--rl.epsilon_start','0.4','--rl.epsilon_end','0.03','--rl.epsilon_decay_steps','15000')
$p = Start-Process -FilePath "python" -ArgumentList $args -WorkingDirectory "D:\IEEE\project" -RedirectStandardOutput "D:\IEEE\project\results\train_robA.txt" -RedirectStandardError "D:\IEEE\project\results\train_robA.err" -PassThru
Write-Output ("PID=" + $p.Id)
