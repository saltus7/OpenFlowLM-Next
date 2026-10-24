# save as run-oflm.ps1 next to build-vs18\
$env:PATH = "C:\Users\user\miniforge3\Library\bin;$env:PATH"
& "$PSScriptRoot\build-vs18\src\oflm.exe" @args
