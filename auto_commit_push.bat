@echo off
:: Lấy ngày giờ chuẩn định dạng YYYY-MM-DD_HH-MM-SS
for /f %%i in ('wmic os get localdatetime ^| find "."') do set dt=%%i

:: Tách thành ngày và giờ
set year=%dt:~0,4%
set month=%dt:~4,2%
set day=%dt:~6,2%
set hour=%dt:~8,2%
set minute=%dt:~10,2%
set second=%dt:~12,2%

:: Tạo commit message tự động
set commit_message=Update %year%-%month%-%day% %hour%:%minute%:%second%

:: In commit message ra màn hình
echo Commit Message: %commit_message%

:: Chạy git
git add .
git commit -m "%commit_message%"
git push

:: Thoát CMD
exit
