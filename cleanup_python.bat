@echo off
echo ========================================
echo Python/Conda 环境清理脚本
echo 请确保以管理员权限运行此脚本
echo ========================================
echo.

REM 检查管理员权限
net session >nul 2>&1
if %errorLevel% == 0 (
    echo [✓] 检测到管理员权限
) else (
    echo [✗] 错误: 需要管理员权限运行此脚本
    echo 请右键选择"以管理员身份运行"
    pause
    exit /b 1
)

echo.
echo [1/6] 停止可能正在运行的Python/Conda进程...
taskkill /f /im python.exe >nul 2>&1
taskkill /f /im pythonw.exe >nul 2>&1
taskkill /f /im conda.exe >nul 2>&1
echo [✓] 进程清理完成

echo.
echo [2/6] 删除Anaconda/Miniconda目录...

REM 删除系统级安装目录
if exist "C:\ProgramData\Anaconda3" (
    echo 删除 C:\ProgramData\Anaconda3...
    rd /s /q "C:\ProgramData\Anaconda3" 2>nul
    if exist "C:\ProgramData\Anaconda3" (
        echo [!] 警告: C:\ProgramData\Anaconda3 可能仍有部分文件未删除
    ) else (
        echo [✓] C:\ProgramData\Anaconda3 删除成功
    )
) else (
    echo [✓] C:\ProgramData\Anaconda3 不存在
)

if exist "C:\Anaconda3" (
    echo 删除 C:\Anaconda3...
    rd /s /q "C:\Anaconda3" 2>nul
    echo [✓] C:\Anaconda3 删除完成
)

if exist "C:\Miniconda3" (
    echo 删除 C:\Miniconda3...
    rd /s /q "C:\Miniconda3" 2>nul
    echo [✓] C:\Miniconda3 删除完成
)

REM 删除用户目录下的conda相关文件
echo 删除用户目录下的conda文件...
if exist "%USERPROFILE%\miniconda3" (
    rd /s /q "%USERPROFILE%\miniconda3" 2>nul
    echo [✓] %USERPROFILE%\miniconda3 删除完成
)

if exist "%USERPROFILE%\Anaconda3" (
    rd /s /q "%USERPROFILE%\Anaconda3" 2>nul
    echo [✓] %USERPROFILE%\Anaconda3 删除完成
)

if exist "%USERPROFILE%\.conda" (
    rd /s /q "%USERPROFILE%\.conda" 2>nul
    echo [✓] %USERPROFILE%\.conda 删除完成
)

if exist "%USERPROFILE%\.condarc" (
    del /f /q "%USERPROFILE%\.condarc" 2>nul
    echo [✓] .condarc 配置文件删除完成
)

REM 删除AppData中的conda缓存
if exist "%LOCALAPPDATA%\conda" (
    rd /s /q "%LOCALAPPDATA%\conda" 2>nul
    echo [✓] 本地conda缓存删除完成
)

if exist "%APPDATA%\conda" (
    rd /s /q "%APPDATA%\conda" 2>nul
    echo [✓] 漫游conda缓存删除完成
)

echo.
echo [3/6] 删除其他Python安装目录...

REM 删除Python Launcher安装
if exist "%LOCALAPPDATA%\Programs\Python" (
    rd /s /q "%LOCALAPPDATA%\Programs\Python" 2>nul
    echo [✓] 用户级Python安装删除完成
)

REM 删除系统级Python安装（常见位置）
for %%v in (37 38 39 310 311 312 313) do (
    if exist "C:\Python%%v" (
        rd /s /q "C:\Python%%v" 2>nul
        echo [✓] Python%%v 删除完成
    )
)

echo.
echo [4/6] 清理注册表项...
echo 删除Python相关注册表项...

REM 删除Python注册表项（用户级）
reg delete "HKCU\Software\Python" /f >nul 2>&1
reg delete "HKCU\Software\Classes\Python.File" /f >nul 2>&1

REM 删除Python注册表项（系统级）
reg delete "HKLM\SOFTWARE\Python" /f >nul 2>&1
reg delete "HKLM\SOFTWARE\WOW6432Node\Python" /f >nul 2>&1

REM 删除Anaconda注册表项
reg delete "HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\Anaconda3" /f >nul 2>&1
reg delete "HKCU\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\Anaconda3" /f >nul 2>&1

echo [✓] 注册表清理完成

echo.
echo [5/6] 清理环境变量...

REM 备份当前PATH
echo 备份当前PATH环境变量...
reg query "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v PATH > "%TEMP%\path_backup_system.txt" 2>nul
reg query "HKCU\Environment" /v PATH > "%TEMP%\path_backup_user.txt" 2>nul
echo [✓] PATH已备份到临时文件

REM 删除Python/Conda相关环境变量
echo 删除Conda相关环境变量...
reg delete "HKCU\Environment" /v CONDA_DEFAULT_ENV /f >nul 2>&1
reg delete "HKCU\Environment" /v CONDA_EXE /f >nul 2>&1
reg delete "HKCU\Environment" /v CONDA_PREFIX /f >nul 2>&1
reg delete "HKCU\Environment" /v CONDA_PROMPT_MODIFIER /f >nul 2>&1
reg delete "HKCU\Environment" /v CONDA_PYTHON_EXE /f >nul 2>&1
reg delete "HKCU\Environment" /v CONDA_SHLVL /f >nul 2>&1

reg delete "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v CONDA_DEFAULT_ENV /f >nul 2>&1
reg delete "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v CONDA_EXE /f >nul 2>&1

echo [✓] 环境变量清理完成

echo.
echo [6/6] 清理开始菜单快捷方式...
if exist "%APPDATA%\Microsoft\Windows\Start Menu\Programs\Anaconda3" (
    rd /s /q "%APPDATA%\Microsoft\Windows\Start Menu\Programs\Anaconda3" 2>nul
    echo [✓] Anaconda开始菜单项删除完成
)

if exist "%ALLUSERSPROFILE%\Microsoft\Windows\Start Menu\Programs\Anaconda3" (
    rd /s /q "%ALLUSERSPROFILE%\Microsoft\Windows\Start Menu\Programs\Anaconda3" 2>nul
    echo [✓] 系统级Anaconda开始菜单项删除完成
)

echo.
echo ========================================
echo 清理完成！
echo ========================================
echo.
echo 重要提醒:
echo 1. 请重启计算机以确保所有更改生效
echo 2. PATH环境变量的备份文件位于:
echo    - 系统PATH: %TEMP%\path_backup_system.txt
echo    - 用户PATH: %TEMP%\path_backup_user.txt
echo 3. 如需安装新的Miniconda，请访问:
echo    https://docs.conda.io/projects/miniconda/en/latest/
echo.

REM 询问是否立即下载Miniconda
set /p download_conda="是否现在下载最新的Miniconda安装程序? (y/n): "
if /i "%download_conda%"=="y" (
    echo.
    echo 正在下载Miniconda安装程序...
    powershell -Command "& {Invoke-WebRequest -Uri 'https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe' -OutFile 'Miniconda3-latest-Windows-x86_64.exe'}"
    if exist "Miniconda3-latest-Windows-x86_64.exe" (
        echo [✓] 下载完成: Miniconda3-latest-Windows-x86_64.exe
        echo 请运行此安装程序来安装全新的Python环境
        set /p run_installer="是否现在运行安装程序? (y/n): "
        if /i "!run_installer!"=="y" (
            start "" "Miniconda3-latest-Windows-x86_64.exe"
        )
    ) else (
        echo [✗] 下载失败，请手动访问官网下载
    )
)

echo.
pause