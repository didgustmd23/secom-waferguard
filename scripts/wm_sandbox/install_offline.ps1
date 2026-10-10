# ==========================================
# Sandbox 내부의 오프라인 검사 환경 준비
# - 설치 파일과 wheel은 읽기 전용 공유 폴더에서 사용
# - 호스트에서 실수로 실행하는 것을 계정 확인으로 방지
# - Python과 패키지만 설치하며 원본 데이터는 읽지 않음
# - 이 계정 확인은 OS 격리의 보안 증명이 아님
# ==========================================
$ErrorActionPreference = 'Stop'

if ($env:USERNAME -ne 'WDAGUtilityAccount') {
    throw 'Windows Sandbox 내부에서만 실행하세요. 호스트에는 설치하지 않습니다.'
}

$installer = 'C:\wm_offline\python-3.13.16-amd64.exe'
$wheelDirectory = 'C:\wm_offline\wheels'
$requirements = 'C:\wm_tools\requirements.txt'
$sandboxPython = 'C:\wm_python\python.exe'
$defaultPython = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python313\python.exe'

# 필수 준비물이 없으면 네트워크로 보완하지 않고 중단한다.
foreach ($requiredPath in @($installer, $wheelDirectory, $requirements)) {
    if (-not (Test-Path -LiteralPath $requiredPath)) {
        throw '오프라인 준비물이 없습니다. 호스트의 준비 폴더와 공유 설정을 확인하세요.'
    }
}

# 서명 유효성은 다운로드 출처 확인을 대신하지 않는다.
if ((Get-AuthenticodeSignature -LiteralPath $installer).Status -ne 'Valid') {
    throw 'Python 설치 파일의 서명이 유효하지 않습니다. 실행을 중단합니다.'
}

# 수동으로 기본 위치에 설치한 경우에도 재설치하지 않고 아래 버전 검사를 수행한다.
if (-not (Test-Path -LiteralPath $sandboxPython)) {
    if (Test-Path -LiteralPath $defaultPython) {
        $sandboxPython = $defaultPython
    } else {
        # 개발 라이브러리·문서·Tcl/Tk를 제외한다. 설치 시간 단축을 보장하지는 않는다.
        Write-Output 'Sandbox 내부에 최소 구성 Python을 설치합니다. 잠시 기다려 주세요.'
        $process = Start-Process -FilePath $installer -ArgumentList @(
            '/quiet', 'InstallAllUsers=0', 'TargetDir=C:\wm_python',
            'Include_pip=1', 'Include_launcher=0', 'Include_test=0', 'PrependPath=0',
            'Include_dev=0', 'Include_doc=0', 'Include_tcltk=0', 'Include_symbols=0'
        ) -WindowStyle Hidden -Wait -PassThru
        if ($process.ExitCode -ne 0) {
            throw 'Python 설치에 실패했습니다. 원본을 읽지 말고 설치 상태를 확인하세요.'
        }
    }
}

# 사용자 pip 설정과 인덱스를 사용하지 않고 로컬 wheel만 설치한다.
& $sandboxPython -I -m pip --isolated install --no-index --no-cache-dir `
    --only-binary=:all: --find-links $wheelDirectory -r $requirements
if ($LASTEXITCODE -ne 0) {
    throw '오프라인 패키지 설치에 실패했습니다. 원본을 읽지 않습니다.'
}

# -I 모드로 작업 폴더의 임의 모듈을 배제하고 import·의존성만 검사한다.
& $sandboxPython -I -m pip --isolated check
if ($LASTEXITCODE -ne 0) { throw '패키지 의존성 검사에 실패했습니다.' }
& $sandboxPython -I -c "import sys, struct, numpy, pandas; assert sys.version_info[:3] == (3, 13, 16); assert struct.calcsize('P') == 8; assert numpy.__version__ == '2.2.6'; assert pandas.__version__ == '2.2.3'"
if ($LASTEXITCODE -ne 0) { throw 'Python 또는 패키지 버전 검사에 실패했습니다.' }

Write-Output '오프라인 검사 환경 준비 완료. 원본은 읽지 않았습니다.'
