@echo off
:: Tên file: push_to_github.bat
:: Tự động commit và push lên GitHub

:: Đặt tiêu đề cửa sổ CMD
title 🚀 Auto Commit and Push to GitHub

:: Lấy ngày giờ hiện tại
for /f "tokens=1-5 delims=/: " %%d in ("%date% %time%") do (
    set datestamp=%%d-%%e-%%f
    set timestamp=%%g-%%h
)

:: Ghép nội dung commit
set commit_message=Auto commit: %datestamp% %timestamp%

:: Thực hiện git add
git add .

:: Commit với thông điệp tự động
git commit -m "%commit_message%"

:: Lấy nhánh hiện tại
for /f "delims=" %%b in ('git rev-parse --abbrev-ref HEAD') do set branch=%%b

:: Push lên nhánh hiện tại
git push origin %branch%

:: Thông báo hoàn tất
echo ✅ Đã push code lên GitHub trên nhánh %branch%.
pause
