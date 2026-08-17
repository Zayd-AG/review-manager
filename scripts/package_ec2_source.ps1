param(
    [string]$OutputPath = "dist\feedback-lens-ec2.zip"
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$outputFullPath = Join-Path $projectRoot $OutputPath
$outputDirectory = Split-Path -Parent $outputFullPath
New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null
Remove-Item -LiteralPath $outputFullPath -Force -ErrorAction SilentlyContinue

$relativeOutputPath = $OutputPath.Replace("\\", "/")
$excludePatterns = @(
    ".git", ".venv", ".uv-cache", ".uv-python", ".hf_cache",
    ".feedback-lens-local", "frontend/node_modules", "frontend/dist",
    "data/raw", "data/processed/pipeline_runs", "finetuning/checkpoints/wandb",
    "finetuning/checkpoints/*/checkpoint-*", ".env", ".env.ec2", $relativeOutputPath
)

$tarArguments = @("-a", "-c", "-f", $outputFullPath)
foreach ($pattern in $excludePatterns) {
    $tarArguments += "--exclude=$pattern"
}
$tarArguments += @("-C", $projectRoot, ".")

& tar.exe @tarArguments
if ($LASTEXITCODE -ne 0) {
    throw "Failed to create deployment archive."
}

Write-Host "Created $outputFullPath"
