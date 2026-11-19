$ErrorActionPreference = "Continue"
$args = @('-u','-m','src.strategy.train_detector','--seed','0')
$p = Start-Process -FilePath "python" -ArgumentList $args -WorkingDirectory "D:\IEEE\project" -RedirectStandardOutput "D:\IEEE\project\results\det_train.txt" -RedirectStandardError "D:\IEEE\project\results\det_train.err" -PassThru
Write-Output ("PID=" + $p.Id)
