$ErrorActionPreference = "Continue"
$args = @('-u','-m','src.train','--rl.replay','partitioned','--rl.strategy_source','groundtruth','--seed','0','--rl.episodes','120','--rl.in_episode_switch','true','--rl.switch_step','100','--rl.episode_len','400')
$p = Start-Process -FilePath "python" -ArgumentList $args -WorkingDirectory "D:\IEEE\project" -RedirectStandardOutput "D:\IEEE\project\results\probe_sw.txt" -RedirectStandardError "D:\IEEE\project\results\probe_sw.err" -PassThru
Write-Output ("PID=" + $p.Id)
