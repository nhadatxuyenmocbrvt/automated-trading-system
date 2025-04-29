@echo off
set /p commit_message="Enter commit message: "

:: Đảm bảo file .env được ignore
if not exist ".gitignore" (
    echo .env>>.gitignore
)

:: Add tất cả file đã cho phép (không add file .env nếu đã ignore)
git add .

:: Hiển thị trạng thái để kiểm tra lần nữa
git status

:: Commit với message
git commit -m "%commit_message%"

:: Push lên remote
git push
