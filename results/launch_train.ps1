$ErrorActionPreference = "Continue"
Remove-Item -Force "D:\IEEE\project\results\train_g.txt","D:\IEEE\project\results\train_g.err" -ErrorAction SilentlyContinue
$p = Start-Process -FilePath "D:\venv\Scripts\python.exe" `
    -ArgumentList '-u','-m','src.train','--rl.replay','partitioned','--rl.episodes','800','--rl.strategy_source','groundtruth','--seed','0','--rl.seed_demos','30' `
    -WorkingDirectory "D:\IEEE\project" `
    -RedirectStandardOutput "D:\IEEE\project\results\train_g.txt" `
    -RedirectStandardError "D:\IEEE\project\results\train_g.err" `
    -PassThru
Write-Output ("PID=" + $p.Id)
