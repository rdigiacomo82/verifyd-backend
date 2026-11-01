# VeriFYD Access — optional model dependency fetch. No admin rights or Node required.
# Run from the extracted project: powershell -ExecutionPolicy Bypass -File .\scripts\Install-MediaPipe-Assets.ps1
# Downloads public Google MediaPipe runtime/model ONLY at install time, to your own extension folder.
$ErrorActionPreference = 'Stop'
$vendor = Join-Path $PSScriptRoot '..\extension\vendor'
$wasm = Join-Path $vendor 'wasm'
New-Item -ItemType Directory -Force -Path $wasm | Out-Null
$base = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22'
$files = @(
  @('vision_bundle.mjs', "$base/vision_bundle.mjs"),
  @('wasm/vision_wasm_internal.js', "$base/wasm/vision_wasm_internal.js"),
  @('wasm/vision_wasm_internal.wasm', "$base/wasm/vision_wasm_internal.wasm"),
  @('wasm/vision_wasm_nosimd_internal.js', "$base/wasm/vision_wasm_nosimd_internal.js"),
  @('wasm/vision_wasm_nosimd_internal.wasm', "$base/wasm/vision_wasm_nosimd_internal.wasm"),
  @('face_landmarker.task', 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task')
)
foreach($entry in $files) {
  $target = Join-Path $vendor $entry[0]
  $url = $entry[1]
  Write-Host "Downloading $($entry[0]) from Google/JSDelivr..."
  Invoke-WebRequest -Uri $url -OutFile $target -UseBasicParsing
  if((Get-Item $target).Length -lt 4096) { throw "Unexpected small file: $target. Verify downloaded URL." }
  $hash = (Get-FileHash -Algorithm SHA256 -Path $target).Hash
  Write-Host "SHA256 $($entry[0]): $hash"
}
Write-Host ''
Write-Host 'Assets present. In Chrome/Edge extensions, Reload VeriFYD Access, then opt in to Start Camera.'
Write-Host 'These files are not downloaded while the extension runs. Camera images stay on your device.'
