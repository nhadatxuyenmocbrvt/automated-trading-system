@echo off
REM Kiểm tra xem .gitignore có tồn tại không, nếu không thì tạo một file để loại trừ các file nhạy cảm
if not exist .gitignore (
    echo Tạo file .gitignore để loại trừ các file nhạy cảm...
    (
        echo .env
        echo *.log
        echo logs/
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
echo Commit với message: "%commit_message%"
git add .
git commit -m "%commit_message%"
git push