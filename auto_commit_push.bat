@echo off
:: Lấy thời gian hiện tại
for /f "tokens=1-4 delims=/ " %%a in ('date /t') do set mydate=%%d-%%b-%%a
for /f "tokens=1-2 delims=: " %%a in ('time /t') do set mytime=%%a:%%b

:: Tạo commit message tự động
set commit_message=Update %mydate% %mytime%

:: Hiển thị commit message
echo Commit Message: %commit_message%

:: Tiến hành git
git add .
git commit -m "%commit_message%"
git push

:: Thoát sau khi hoàn thành
exit