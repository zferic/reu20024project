# pwsh script to clean up papers directory
$papersDir = "papers"

Write-Host "Starting cleanup of papers directory..."

function Test-HasEmptySections {
    param($filePath)
    
    $content = Get-Content $filePath -Raw
    $sections = @(
        "### Abstract ###",
        "### Introduction ###",
        "### Methods ###",
        "### Results ###",
        "### Conclusion ###"
    )
    
    foreach ($section in $sections) {
        if ($content -match "$section\s*(?:###|$)") {
            return $true
        }
    }
    return $false
}

# Get all text files
$files = Get-ChildItem -Path $papersDir -Filter "*.txt"

# Group files by their base name (without _1, _2, _3 suffixes)
$groupedFiles = $files | Group-Object { $_.Name -replace '_[0-9]+\.txt$','.txt' }

foreach ($group in $groupedFiles) {
    Write-Host "Processing group: $($group.Name)"
    
    # Keep only the file without suffix if it exists, otherwise keep the one with _1
    $filesToKeep = $group.Group | Where-Object { $_.Name -notmatch '_[0-9]+\.txt$' }
    if (-not $filesToKeep) {
        $filesToKeep = $group.Group | Where-Object { $_.Name -match '_1\.txt$' }
    }
    
    # Remove other files in the group
    $filesToRemove = $group.Group | Where-Object { $_ -notin $filesToKeep }
    foreach ($file in $filesToRemove) {
        Write-Host "Removing duplicate: $($file.Name)"
        Remove-Item $file.FullName
    }
    
    # Check remaining file for empty sections
    foreach ($file in $filesToKeep) {
        if (Test-HasEmptySections $file.FullName) {
            Write-Host "Removing file with empty sections: $($file.Name)"
            Remove-Item $file.FullName
        }
    }
}

# Clean up associated JSON files that no longer have a corresponding txt file
$jsonFiles = Get-ChildItem -Path $papersDir -Filter "*.json"
foreach ($jsonFile in $jsonFiles) {
    $correspondingTxt = $jsonFile.Name -replace '_tables\.json$','.txt'
    $correspondingTxt = $correspondingTxt -replace '_[0-9]+_tables\.json$','.txt'
    if (-not (Test-Path (Join-Path $papersDir $correspondingTxt))) {
        Write-Host "Removing orphaned JSON file: $($jsonFile.Name)"
        Remove-Item $jsonFile.FullName
    }
}

Write-Host "Cleanup complete!" 