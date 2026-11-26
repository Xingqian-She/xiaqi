[CmdletBinding()]
param(
    [Parameter(Mandatory, Position = 0)]
    [string]$Experiment,
    [switch]$New,
    [switch]$Resume,
    [switch]$Shell
)

$ErrorActionPreference = 'Stop'
if (@($New, $Resume, $Shell).Where({ $_.IsPresent }).Count -ne 1) {
    throw 'Select exactly one mode: -New, -Resume or -Shell.'
}
if ([string]::IsNullOrWhiteSpace($Experiment) -or $Experiment.EndsWith('.') -or
    $Experiment -ne $Experiment.Trim() -or $Experiment.IndexOfAny([IO.Path]::GetInvalidFileNameChars()) -ge 0) {
    throw 'The experiment name must be a single directory name.'
}

$repoDirectory = Split-Path -Parent $PSScriptRoot
$experimentDirectory = Join-Path $repoDirectory "runs/$Experiment"
$databaseDirectory = Join-Path $experimentDirectory 'postgresql'
if (-not (Test-Path -LiteralPath (Join-Path $experimentDirectory 'config.toml') -PathType Leaf)) {
    throw "Experiment configuration not found in $experimentDirectory."
}
$checkpoints = Join-Path $experimentDirectory 'checkpoints'
$hasCheckpoint = $false
if (Test-Path -LiteralPath $checkpoints -PathType Container) {
    $hasCheckpoint = [bool](Get-ChildItem -LiteralPath $checkpoints -File -Filter 'step-*.pt' |
        Where-Object { $_.Name -cmatch '^step-[0-9]+\.pt$' } | Select-Object -First 1)
}
if ($New -and $hasCheckpoint) {
    throw 'The experiment already has checkpoints. Use -Resume.'
}
$requireExisting = $Resume -or $hasCheckpoint
if ($requireExisting -and -not (Test-Path -LiteralPath $databaseDirectory -PathType Container)) {
    throw "Restore the existing experiment database at $databaseDirectory."
}
if (-not $requireExisting) {
    [void][IO.Directory]::CreateDirectory($databaseDirectory)
}

$containerDirectory = "/workspace/runs/$Experiment"
$composeArgs = @('run', '--rm', '--no-deps', '--entrypoint', 'python',
    '--volume', "${databaseDirectory}:/var/lib/postgresql",
    '--env', "XIAQI_EXPERIMENT_DIR=$containerDirectory")
if (-not $Shell) {
    $composeArgs += '-T'
}
$composeArgs += @('dev', '/workspace/docker/dev-entrypoint.py')
$databaseMode = 'init'
if ($requireExisting) {
    $composeArgs += '--existing'
    $databaseMode = 'existing'
}
$composeArgs += @('bash', 'src/data/postgresql/bootstrap.sh', $databaseMode)
if ($Shell) {
    $composeArgs += 'bash'
} else {
    $composeArgs += @('bash', 'scripts/train.sh', "$containerDirectory/config.toml")
    if ($Resume) {
        $composeArgs += '--resume'
    }
}

docker compose -f (Join-Path $repoDirectory 'compose.yaml') @composeArgs
exit $LASTEXITCODE
