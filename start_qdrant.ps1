# Start Qdrant Docker Container
$currentDir = Get-Location
$storagePath = "$($currentDir.Path)\qdrant_storage"

# Create storage directory if it doesn't exist
if (-not (Test-Path -Path $storagePath)) {
    New-Item -ItemType Directory -Path $storagePath | Out-Null
}

Write-Host "Starting Qdrant with storage at: $storagePath"

docker run -d -p 6333:6333 -p 6334:6334 `
    -v "$($storagePath):/qdrant/storage" `
    --name bato-qdrant `
    qdrant/qdrant

Write-Host "Qdrant started on port 6333"
