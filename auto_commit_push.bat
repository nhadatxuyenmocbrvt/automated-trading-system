@echo off
:: Tên file: auto_commit_push.bat
:: Auto commit and push, exclude sensitive files

:: Đặt tiêu đề CMD
title 🚀 Auto Commit and Push to GitHub (Exclude .env & master.key)

:: Lấy ngày giờ commit
for /f "tokens=1-5 delims=/: " %%d in ("%date% %time%") do (
    set datestamp=%%d-%%e-%%f
    set timestamp=%%g-%%h
)

:: Ghép commit message
set commit_message=Auto commit: %datestamp% %timestamp%

:: Thêm tất cả thay đổi trừ file nhạy cảm
git add --all
git reset HEAD .env
git reset HEAD master.key

:: Commit với thông điệp
git commit -m "%commit_message%"

:: Lấy nhánh hiện tại
for /f "delims=" %%b in ('git rev-parse --abbrev-ref HEAD') do set branch=%%b

:: Push lên GitHub
git push origin %branch%

:: Thông báo hoàn tất
echo ✅ Đã push lên GitHub (đã bỏ qua .env và master.key)!
pause
