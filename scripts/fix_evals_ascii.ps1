$ErrorActionPreference = "Stop"

$path = "c:\Users\abbas\source\repos\energy-evals\data\evals_full_dataset_with_answers.csv"
if (!(Test-Path $path)) { throw "File not found: $path" }

$backup = "$path.bak_" + (Get-Date -Format "yyyyMMdd_HHmmss")
Copy-Item -LiteralPath $path -Destination $backup -Force
Write-Host "Backup created: $backup"

$bytes = [System.IO.File]::ReadAllBytes($path)

$utf8Strict = [System.Text.Encoding]::GetEncoding(
  "utf-8",
  [System.Text.EncoderFallback]::ExceptionFallback,
  [System.Text.DecoderFallback]::ExceptionFallback
)

try { $text = $utf8Strict.GetString($bytes) }
catch { $text = [System.Text.Encoding]::GetEncoding(1252).GetString($bytes) }

function Get-WeirdScore([string]$s) {
  $m1 = [regex]::Matches($s, "[^\x00-\x7F]").Count
  return $m1
}

function Try-FixMojibake([string]$s) {
  $latin1 = [System.Text.Encoding]::GetEncoding("ISO-8859-1")
  $best = $s
  $bestScore = Get-WeirdScore $s
  for ($i = 0; $i -lt 3; $i++) {
    $candidate = [System.Text.Encoding]::UTF8.GetString($latin1.GetBytes($best))
    $score = Get-WeirdScore $candidate
    if ($score -lt $bestScore) { $best = $candidate; $bestScore = $score } else { break }
  }
  return $best
}

$text = Try-FixMojibake $text

# Known bad 3-char mojibake sequences built from code points (ASCII-only script source)
$badApos1 = ([string][char]0x0101) + ([string][char]0x20AC) + ([string][char]0x2122) # ā€™
$badDash1 = ([string][char]0x0101) + ([string][char]0x20AC) + ([string][char]0x2013) # ā€“
$badApos2 = ([string][char]0x00E2) + ([string][char]0x20AC) + ([string][char]0x2122) # â€™
$badDash2 = ([string][char]0x00E2) + ([string][char]0x20AC) + ([string][char]0x201C) # often bad dash/quote variants

$text = $text.Replace($badApos1, "'")
$text = $text.Replace($badDash1, "-")
$text = $text.Replace($badApos2, "'")
$text = $text.Replace($badDash2, "-")
$text = $text.Replace("MW?weighted", "MW-weighted")

# Normalize common unicode punctuation to ASCII
$text = $text.Replace([string][char]0x2018, "'")
$text = $text.Replace([string][char]0x2019, "'")
$text = $text.Replace([string][char]0x201C, '"')
$text = $text.Replace([string][char]0x201D, '"')
$text = $text.Replace([string][char]0x2013, "-")
$text = $text.Replace([string][char]0x2014, "-")
$text = $text.Replace([string][char]0x2026, "...")
$text = $text.Replace([string][char]0x2022, "-")
$text = $text.Replace([string][char]0x00A0, " ")
$text = $text.Replace([string][char]0xFFFD, " ")

$text = [regex]::Replace($text, "[\x00-\x08\x0B\x0C\x0E-\x1F]", "")
$text = $text.Normalize([Text.NormalizationForm]::FormKD)
$text = [regex]::Replace($text, "\p{M}", "")
$text = [regex]::Replace($text, "[^\x00-\x7F]", "")
$text = [regex]::Replace($text, " {2,}", " ")

$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($path, $text, $utf8NoBom)

$nonAscii = Select-String -Path $path -Pattern "[^\x00-\x7F]"
$badCtrl  = Select-String -Path $path -Pattern "[\x00-\x08\x0B\x0C\x0E-\x1F]"

if (($nonAscii.Count -eq 0) -and ($badCtrl.Count -eq 0)) {
  Write-Host "OK: file is ASCII-clean and control-char clean."
} else {
  Write-Host "Still found issues."
  if ($nonAscii.Count -gt 0) {
    Write-Host "Non-ASCII lines:"
    $nonAscii | Select-Object -First 30 | ForEach-Object { "{0}: {1}" -f $_.LineNumber, $_.Line }
  }
  if ($badCtrl.Count -gt 0) {
    Write-Host "Control-char lines:"
    $badCtrl | Select-Object -First 30 | ForEach-Object { "{0}: {1}" -f $_.LineNumber, $_.Line }
  }
}
