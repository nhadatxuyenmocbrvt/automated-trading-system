@echo off
echo ===== Tự động commit và push =====

REM Kiểm tra xem .gitignore có tồn tại không, nếu không thì tạo một file để loại trừ các file nhạy cảm
if not exist .gitignore (
    echo Tạo file .gitignore để loại trừ các file nhạy cảm...
    (
        echo .env
        echo config/security_config.py
    ) > .gitignore
)

REM Tự động tạo message commit với timestamp
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "YYYY=%dt:~0,4%"
set "MM=%dt:~4,2%"
set "DD=%dt:~6,2%"
set "HH=%dt:~8,2%"
set "Min=%dt:~10,2%"
set "commit_message=Cập nhật %YYYY%-%MM%-%DD% %HH%:%Min%"

REM Thêm, commit và push
echo.
echo Đang thêm các file đã thay đổi...
git add .
if %errorlevel% neq 0 (
    echo Lỗi khi thêm file!
    goto end
)

echo.
echo Commit với message: "%commit_message%"
git commit -m "%commit_message%"
if %errorlevel% neq 0 (
    echo Lỗi khi commit!
    goto end
)

echo.
echo Đang push lên repository...
git push
if %errorlevel% neq 0 (
    echo Lỗi khi push!
    goto end
)

echo.
echo ===== Hoàn thành commit và push thành công! =====

:end
echo.
pause