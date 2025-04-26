@echo off
git add .
git commit -m "Auto commit %date% %time%"
git push origin main
echo Auto commit va push thanh cong!
pause
