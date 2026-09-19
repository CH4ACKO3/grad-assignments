[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Source,
    [ValidateSet('pdflatex', 'xelatex', 'lualatex')][string]$Engine,
    [switch]$ShellEscape
)
$ErrorActionPreference = 'Stop'
$repoRoot = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$sourceFile = (Resolve-Path -LiteralPath $Source).Path
if (-not $sourceFile.StartsWith($repoRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase) -or [IO.Path]::GetExtension($sourceFile) -ne '.tex') {
    throw 'Source must be a .tex file inside this worktree.'
}
$sourceDirectory = Split-Path -Parent $sourceFile
$arguments = @('-synctex=1', '-interaction=nonstopmode', '-halt-on-error', '-outdir=tex')
if ($Engine) {
    $arguments += @{pdflatex='-pdf'; xelatex='-xelatex'; lualatex='-lualatex'}[$Engine]
}
if ($ShellEscape) { $arguments += '-shell-escape' }
$arguments += (Split-Path -Leaf $sourceFile)
Push-Location -LiteralPath $sourceDirectory
try {
    & latexmk @arguments
    if ($LASTEXITCODE -ne 0) { throw "LaTeX build failed (exit $LASTEXITCODE)." }
} finally { Pop-Location }
