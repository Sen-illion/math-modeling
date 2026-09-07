param([switch]$Force)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$manifestPath = Join-Path $repoRoot 'skills\skill_manifest.json'
$localDir = Join-Path $repoRoot '.local'
$statePath = Join-Path $localDir 'codex_environment.json'
if (-not (Test-Path -LiteralPath $manifestPath)) { Write-Error "Missing skill manifest: $manifestPath"; exit 2 }
$manifest = Get-Content -Raw -Encoding UTF8 -LiteralPath $manifestPath | ConvertFrom-Json
$manifestHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $manifestPath).Hash
if (-not $Force -and (Test-Path -LiteralPath $statePath)) {
  try {
    $oldState = Get-Content -Raw -Encoding UTF8 -LiteralPath $statePath | ConvertFrom-Json
    if ($oldState.status -eq 'PASS' -and $oldState.manifest_sha256 -eq $manifestHash) {
      Write-Output '[PASS] Cold-start check already passed for the current manifest.'
      Write-Output "Verified at: $($oldState.verified_at)"
      Write-Output 'Use -Force to run the scan again.'
      exit 0
    }
  } catch { Write-Warning 'Previous verification record unreadable; scanning again.' }
}
$gitStatus = 'NOT_A_REPOSITORY'; $gitBranch = $null; $gitRemote = $null
if (Test-Path -LiteralPath (Join-Path $repoRoot '.git')) {
  $gitStatus = 'OK'; $gitBranch = (git -C $repoRoot branch --show-current 2>$null).Trim(); $gitRemote = (git -C $repoRoot remote get-url origin 2>$null).Trim()
}
$candidateRoots = @(); $pluginCacheRoot = Join-Path $env:USERPROFILE '.codex\plugins\cache\mathmodeling-skills\mathmodeling-skills'
if (Test-Path -LiteralPath $pluginCacheRoot) { Get-ChildItem -LiteralPath $pluginCacheRoot -Directory | ForEach-Object { $candidateRoots += $_.FullName } }
$projectPluginRoot = Join-Path $repoRoot '.agents\plugins\mathmodeling-skills'
if (Test-Path -LiteralPath $projectPluginRoot) { $candidateRoots += $projectPluginRoot }
$selectedRoot = $null; $discoveredVersion = $null; $repositoryMatch = $false; $missingSkills = @()
foreach ($candidate in $candidateRoots) {
  $pluginJson = Join-Path $candidate '.codex-plugin\plugin.json'; $skillsRoot = Join-Path $candidate 'skills'
  if (-not (Test-Path -LiteralPath $pluginJson) -or -not (Test-Path -LiteralPath $skillsRoot)) { continue }
  $plugin = Get-Content -Raw -Encoding UTF8 -LiteralPath $pluginJson | ConvertFrom-Json
  if ($plugin.name -ne $manifest.name) { continue }
  $currentMissing = @(); foreach ($skillName in $manifest.required_skill_directories) { if (-not (Test-Path -LiteralPath (Join-Path $skillsRoot "$skillName\SKILL.md"))) { $currentMissing += $skillName } }
  $selectedRoot = $candidate; $discoveredVersion = [string]$plugin.version; $repositoryMatch = ([string]$plugin.repository -eq [string]$manifest.source); $missingSkills = $currentMissing
  if ($repositoryMatch -and $discoveredVersion -eq [string]$manifest.plugin_version -and $missingSkills.Count -eq 0) { break }
}
$skillFound = $null -ne $selectedRoot; $versionMatch = $skillFound -and ($discoveredVersion -eq [string]$manifest.plugin_version); $skillPass = $skillFound -and $versionMatch -and $repositoryMatch -and $missingSkills.Count -eq 0; $overallPass = ($gitStatus -eq 'OK') -and $skillPass; $status = if ($overallPass) { 'PASS' } else { 'FAIL' }
New-Item -ItemType Directory -Path $localDir -Force | Out-Null
$state = [ordered]@{ schema_version=1; status=$status; verified_at=(Get-Date).ToString('o'); machine_local=$true; manifest_sha256=$manifestHash; required_source=$manifest.source; required_ref=$manifest.ref; required_plugin_version=$manifest.plugin_version; git=[ordered]@{status=$gitStatus; branch=$gitBranch; origin=$gitRemote}; skill=[ordered]@{found=$skillFound; path=$selectedRoot; discovered_version=$discoveredVersion; version_match=$versionMatch; repository_match=$repositoryMatch; missing_required_skills=$missingSkills; exact_commit_verifiable_from_cache=$false} }
$state | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 -LiteralPath $statePath
Write-Output "Git repository: $gitStatus"; if ($gitBranch) { Write-Output "Current branch: $gitBranch" }; if ($gitRemote) { Write-Output "Origin: $gitRemote" }; Write-Output ("Mathmodeling skill: " + $(if ($skillFound) { 'FOUND' } else { 'NOT FOUND' })); if ($skillFound) { Write-Output "Install path: $selectedRoot"; Write-Output "Discovered version: $discoveredVersion; required: $($manifest.plugin_version)" }
if ($overallPass) { Write-Output '[PASS] Cold-start check passed. You may begin reading the contest problem.'; exit 0 }
Write-Output '[FAIL] Cold-start check failed. Resolve the reported issues before modeling:'
if ($gitStatus -ne 'OK') { Write-Output '- The current directory is not a valid Git repository.' }
if (-not $skillFound) { Write-Output "- Required skill was not found. Install $($manifest.name) from $($manifest.source) at ref $($manifest.ref)." } else { if (-not $repositoryMatch) { Write-Output '- Repository mismatch.' }; if (-not $versionMatch) { Write-Output "- Version mismatch: found $discoveredVersion; required $($manifest.plugin_version)." }; if ($missingSkills.Count -gt 0) { Write-Output "- Missing skill directories: $($missingSkills -join ', ')" } }
Write-Output "Detailed state written to: $statePath"; exit 2
