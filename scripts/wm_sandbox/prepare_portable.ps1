# ==========================================
# 설치 프로그램 없이 Sandbox 전용 Python 준비
# - 공식 embeddable ZIP과 미리 받은 고정 버전 wheel만 사용
# - 네트워크·pip·Windows 설치 프로그램을 사용하지 않음
# - 압축 경로를 검증하고 별도 실행 폴더에만 압축 해제
# - NumPy·pandas import 성공 전에는 원본을 읽지 않음
# ==========================================
$ErrorActionPreference = 'Stop'
if ($env:USERNAME -ne 'WDAGUtilityAccount') { throw 'Sandbox 내부에서 실행하세요.' }
Add-Type -AssemblyName System.IO.Compression.FileSystem

function Expand-CheckedArchive([string]$ArchivePath, [string]$Destination) {
    # ZIP 내부의 상위 경로·절대 경로·대체 데이터 스트림을 거부한다.
    $root = [IO.Path]::GetFullPath($Destination).TrimEnd('\') + '\'
    $archive = [IO.Compression.ZipFile]::OpenRead($ArchivePath)
    try {
        $totalBytes = 0L
        foreach ($entry in $archive.Entries) {
            $name = $entry.FullName.Replace('/', '\')
            $target = [IO.Path]::GetFullPath((Join-Path $root $name))
            $totalBytes += $entry.Length
            if ($name.Contains(':') -or [IO.Path]::IsPathRooted($name) -or
                -not $target.StartsWith($root, [StringComparison]::OrdinalIgnoreCase) -or
                $totalBytes -gt 1GB) {
                throw '압축 파일의 경로 또는 해제 크기 검증에 실패했습니다.'
            }
        }
    } finally { $archive.Dispose() }
    [IO.Compression.ZipFile]::ExtractToDirectory($ArchivePath, $Destination)
}

$pythonArchive = 'C:\wm_offline\python-3.13.16-embed-amd64.zip'
$wheelNames = @(
    'numpy-2.2.6-cp313-cp313-win_amd64.whl',
    'pandas-2.2.3-cp313-cp313-win_amd64.whl',
    'python_dateutil-2.9.0.post0-py2.py3-none-any.whl',
    'pytz-2025.2-py2.py3-none-any.whl',
    'six-1.17.0-py2.py3-none-any.whl',
    'tzdata-2025.2-py2.py3-none-any.whl'
)

# 준비물 누락은 실행 폴더를 만들기 전에 확인한다.
foreach ($required in @($pythonArchive, 'C:\wm_tools\python313._pth')) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "준비 파일이 없습니다: $required"
    }
}
foreach ($wheelName in $wheelNames) {
    if (-not (Test-Path -LiteralPath (Join-Path 'C:\wm_offline\wheels' $wheelName))) {
        throw "준비 wheel이 없습니다: $wheelName"
    }
}

# 실패한 준비 폴더를 삭제하거나 덮어쓰지 않고 새 시도 폴더를 사용한다.
$runtime = Join-Path 'C:\wm_portable' ([Guid]::NewGuid().ToString('N'))
$packages = Join-Path $runtime 'Lib\site-packages'
New-Item -ItemType Directory -Path $runtime -Force | Out-Null
Write-Output '[1/3] Python ZIP 압축 해제'
Expand-CheckedArchive $pythonArchive $runtime
New-Item -ItemType Directory -Path $packages -Force | Out-Null
Write-Output '[2/3] 고정 버전 라이브러리 준비'
foreach ($wheelName in $wheelNames) {
    Write-Output "  $wheelName"
    Expand-CheckedArchive (Join-Path 'C:\wm_offline\wheels' $wheelName) $packages
}
Copy-Item -LiteralPath 'C:\wm_tools\python313._pth' -Destination $runtime

# 실제 import와 버전·64비트를 검증한 실행 파일만 검사기에 전달한다.
Write-Output '[3/3] Python / NumPy / pandas 실행 확인'
$pythonExe = Join-Path $runtime 'python.exe'
& $pythonExe -I -c "import sys, struct, numpy, pandas; assert sys.version_info[:3] == (3,13,16); assert struct.calcsize('P') == 8; assert numpy.__version__ == '2.2.6'; assert pandas.__version__ == '2.2.3'"
if ($LASTEXITCODE -ne 0) { throw '라이브러리 실행 확인 실패. 원본은 읽지 않았습니다.' }
[IO.File]::WriteAllText('C:\wm_portable\active_python.txt', $pythonExe, [Text.Encoding]::ASCII)
Write-Output '준비 완료. 원본은 읽지 않았습니다. 이제 run_inspection.ps1을 실행하세요.'
