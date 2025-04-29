@echo off
echo ===== Tu Dong commit va push =====

REM Kiem tra xem .gitignore co ton tai khong, neu không thi tao mot file đe loai tru cac file nhay cam
if not exist .gitignore (
    echo Tao file .gitignore đe loai tru cac file nhay cam...
    (
        echo .env
        echo config/security_config.py
    ) > .gitignore
)

REM Tu đong tao message commit với timestamp
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "YYYY=%dt:~0,4%"
set "MM=%dt:~4,2%"
set "DD=%dt:~6,2%"
set "HH=%dt:~8,2%"
set "Min=%dt:~10,2%"
set "commit_message=Cap nhat %YYYY%-%MM%-%DD% %HH%:%Min%"

REM Them, commit và push
echo.
echo Đang them cac file da thay doi...
git add .
if %errorlevel% neq 0 (
    echo Loi khi them file!
    goto end
)

echo.
echo Commit voi message: "%commit_message%"
git commit -m "%commit_message%"
if %errorlevel% neq 0 (
    echo Loi khi commit!
    goto end
)

echo.
echo Đang push len repository...
git push
if %errorlevel% neq 0 (
    echo Lỗi khi push!
    goto end
)

echo.
echo ===== Hoan Thanh commit va push Thanh Cong! =====

:end
echo.
pause